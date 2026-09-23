"""ONNX Runtime SCRFD detector + ArcFace embedder (BUILD_SPEC §7.10). **OPT-IN**

Two models, two jobs: SCRFD finds faces and five landmarks per face; ArcFace
turns an aligned 112x112 crop into a 512-d embedding for cosine matching
against the watchlist (faces.py). Neither is instantiated unless
``faces.enabled`` is true — see faces.py's module docstring for why that
matters more here than anywhere else in the system (P6).

**SCRFD's output shape is not obvious and is worth writing down once.** For a
640x640 input this model emits nine tensors: three feature-pyramid levels
(stride 8, 16, 32), each contributing a score tensor, a box tensor and a
landmark tensor. Boxes and landmarks are predicted as *distances from an
anchor point*, not as absolute coordinates — this is the standard SCRFD/FCOS
decode, reproduced faithfully from the reference implementation rather than
approximated, because a subtly wrong anchor grid produces boxes that look
plausible and are off by half a face width.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..faces import FaceConfig, FaceDetection
from . import DetectorConfig, gpu_run_lock, resolve_execution_providers
from .onnx_weapon import nms

logger = logging.getLogger(__name__)

__all__ = [
    "ARCFACE_TEMPLATE_112",
    "MockFaceDetector",
    "MockFaceEmbedder",
    "OnnxFaceDetector",
    "OnnxFaceEmbedder",
    "align_face",
    "decode_scrfd",
]

#: The five reference points (left eye, right eye, nose, left mouth corner,
#: right mouth corner) that a 112x112 ArcFace input is aligned to. Fixed by
#: the model's own training data -- changing these silently degrades every
#: embedding the same way a wrong camera calibration silently degrades speed.
ARCFACE_TEMPLATE_112: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2  # matches this bundled model's output counts; see docstring


def _anchor_centers(height: int, width: int, stride: int, num_anchors: int) -> np.ndarray:
    """Grid of anchor centres for one pyramid level, tiled per anchor.

    Reproduced from the reference SCRFD post-processing rather than derived
    from first principles: the ``[::-1]`` and the tile order are exactly what
    the exported weights expect, and a plausible-looking alternative (e.g.
    swapping x/y) decodes to boxes that are wrong in a way that is not obvious
    from a single test image.
    """
    grid_y, grid_x = np.mgrid[:height, :width]
    centers = np.stack([grid_x, grid_y], axis=-1).astype(np.float32) * stride
    centers = centers.reshape(-1, 2)
    if num_anchors > 1:
        centers = np.repeat(centers, num_anchors, axis=0)
    return centers


def decode_scrfd(
    outputs: list[np.ndarray],
    input_size: tuple[int, int],
    *,
    min_conf: float = 0.60,
    nms_iou: float = 0.45,
    strides: tuple[int, ...] = _STRIDES,
    num_anchors: int = _NUM_ANCHORS,
) -> list[FaceDetection]:
    """Decode SCRFD's nine raw tensors into faces, in MODEL-space coordinates.

    ``outputs`` must be grouped BY TYPE, not by stride: all three score
    tensors first, then all three box tensors, then all three landmark
    tensors, each group ordered by ascending stride. That is the order
    ``ort.InferenceSession.run`` returns them in for this export (confirmed
    against the actual model, not assumed) -- a model with a different head
    count or grouping raises rather than silently decoding garbage.
    """
    expected = len(strides) * 3
    if len(outputs) != expected:
        raise ValueError(
            f"expected {expected} SCRFD output tensors ({len(strides)} strides x "
            f"score/bbox/kps), got {len(outputs)}. This export does not match "
            f"the decoder's assumptions."
        )
    n = len(strides)
    score_outputs = outputs[0:n]
    bbox_outputs = outputs[n : 2 * n]
    kps_outputs = outputs[2 * n : 3 * n]

    w, h = input_size
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    kps_list: list[np.ndarray] = []

    for level, stride in enumerate(strides):
        score = score_outputs[level][:, 0]
        bbox_dist = bbox_outputs[level] * stride
        kps_dist = kps_outputs[level] * stride

        fh, fw = h // stride, w // stride
        centers = _anchor_centers(fh, fw, stride, num_anchors)
        if centers.shape[0] != score.shape[0]:
            raise ValueError(
                f"stride {stride}: anchor grid has {centers.shape[0]} points but "
                f"the model emitted {score.shape[0]} scores -- input_size "
                f"{input_size} does not match what this export was traced for."
            )

        keep = score >= min_conf
        if not keep.any():
            continue
        c = centers[keep]
        d = bbox_dist[keep]
        box = np.stack(
            [c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1
        )
        kd = kps_dist[keep]
        kp = np.stack([c[:, i % 2] + kd[:, i] for i in range(kd.shape[1])], axis=1)

        boxes.append(box)
        scores.append(score[keep])
        kps_list.append(kp)

    if not boxes:
        return []

    all_boxes = np.concatenate(boxes, axis=0)
    all_scores = np.concatenate(scores, axis=0)
    all_kps = np.concatenate(kps_list, axis=0)

    keep_idx = nms(all_boxes, all_scores, nms_iou)
    faces = []
    for i in keep_idx:
        pts = all_kps[i].reshape(-1, 2)
        b = all_boxes[i]
        faces.append(
            FaceDetection(
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(all_scores[i]),
                landmarks=tuple((float(x), float(y)) for x, y in pts),
            )
        )
    return faces


def align_face(image: np.ndarray, landmarks: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Warp a face to the fixed 112x112 ArcFace template using its 5 landmarks.

    Alignment, not just cropping, is what makes the embedding meaningful: a
    tilted or off-centre face embeds as a worse match to its own enrolled
    photo than a genuinely different, well-aligned face would. Raises if fewer
    than 5 landmarks are available rather than falling back to a plain crop,
    because a plain crop would not fail loudly -- it would just quietly embed
    badly.
    """
    import cv2

    if len(landmarks) != 5:
        raise ValueError(f"face alignment needs 5 landmarks, got {len(landmarks)}")
    src = np.array(landmarks, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE_112, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("could not estimate an alignment transform from these landmarks")
    return cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)


class OnnxFaceDetector:
    """SCRFD session. Only constructed when ``faces.enabled`` is true."""

    def __init__(self, cfg: DetectorConfig, face_cfg: FaceConfig) -> None:
        self.cfg = cfg
        self.face_cfg = face_cfg
        weights = Path(cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"face detector weights not found: {weights}\n"
                f"Run `make models` to download them. See docs/MODELS.md."
            )

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = resolve_execution_providers(cfg, ort.get_available_providers())
        try:
            self._session = ort.InferenceSession(str(weights), options, providers=providers)
        except Exception:
            logger.exception(
                "could not create a face-detector session with providers %s; retrying on CPU",
                [p[0] if isinstance(p, tuple) else p for p in providers],
            )
            self._session = ort.InferenceSession(
                str(weights), options, providers=["CPUExecutionProvider"]
            )
        self._input_name = self._session.get_inputs()[0].name
        self._run_lock = gpu_run_lock(self._session)
        logger.info(
            "onnx face detector loaded weights=%s providers=%s",
            weights,
            self._session.get_providers(),
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self.cfg.input_size

    def detect(self, letterboxed: np.ndarray) -> list[FaceDetection]:
        w, h = self.cfg.input_size
        if letterboxed.shape[:2] != (h, w):
            raise ValueError(
                f"face detector input must be pre-letterboxed to {w}x{h}, got "
                f"{letterboxed.shape[1]}x{letterboxed.shape[0]}"
            )
        # SCRFD's own preprocessing (matches cv2.dnn.blobFromImage(1/128, ...,
        # mean=(127.5,)*3) from the reference implementation): BGR->RGB, then
        # (x - 127.5) / 128. Feeding raw 0..255 values -- easy to miss, since
        # the model still runs and still produces plausible-looking scores --
        # measurably degrades both confidence and localisation; see the
        # module's test for the gap this closes.
        blob = letterboxed.transpose(2, 0, 1)[::-1].astype(np.float32)
        blob = (blob - 127.5) / 128.0
        with self._run_lock:
            outputs = self._session.run(None, {self._input_name: blob[None]})
        return decode_scrfd(
            outputs,
            self.cfg.input_size,
            min_conf=self.face_cfg.detect_conf,
        )


class OnnxFaceEmbedder:
    """ArcFace (w600k_mbf) session. Input is a pre-aligned 112x112 crop."""

    def __init__(self, cfg: DetectorConfig) -> None:
        weights = Path(cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"face embedder weights not found: {weights}\n"
                f"Run `make models` to download them. See docs/MODELS.md."
            )

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = resolve_execution_providers(cfg, ort.get_available_providers())
        try:
            self._session = ort.InferenceSession(str(weights), options, providers=providers)
        except Exception:
            logger.exception("could not create a face-embedder session; retrying on CPU")
            self._session = ort.InferenceSession(
                str(weights), options, providers=["CPUExecutionProvider"]
            )
        self._input_name = self._session.get_inputs()[0].name
        self._run_lock = gpu_run_lock(self._session)
        logger.info("onnx face embedder loaded weights=%s", weights)

    def embed(self, aligned_112: np.ndarray) -> tuple[float, ...]:
        if aligned_112.shape[:2] != (112, 112):
            raise ValueError(f"embedder input must be 112x112, got {aligned_112.shape[:2]}")
        # ArcFace's own preprocessing: BGR->RGB, (x - 127.5) / 128.
        blob = aligned_112.transpose(2, 0, 1)[::-1].astype(np.float32)
        blob = (blob - 127.5) / 128.0
        with self._run_lock:
            outputs = self._session.run(None, {self._input_name: blob[None]})
        return tuple(float(v) for v in outputs[0][0])


class MockFaceDetector:
    """Scripted faces, no weights. "Mock the GPU, not the logic.\" """

    def __init__(self, results: list[list[FaceDetection]] | None = None) -> None:
        self.results = list(results or [])
        self.calls = 0
        self.input_size = (640, 640)

    def detect(self, letterboxed: np.ndarray) -> list[FaceDetection]:
        self.calls += 1
        if not self.results:
            return []
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class MockFaceEmbedder:
    def __init__(self, embeddings: list[tuple[float, ...]] | None = None) -> None:
        self.embeddings = list(embeddings or [])
        self.calls = 0

    def embed(self, aligned_112: np.ndarray) -> tuple[float, ...]:
        self.calls += 1
        if not self.embeddings:
            return (0.0,) * 512
        return self.embeddings[min(self.calls - 1, len(self.embeddings) - 1)]
