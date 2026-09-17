"""ONNX Runtime pose estimator for hand signals (BUILD_SPEC §7.7, gestures).

Second-stage, crop-driven, exactly like ANPR (§7.9) and for the same reasons:

* It costs **nothing** when disabled or when no person is being tracked. The
  primary detector's hot path is untouched, so turning gestures on cannot slow
  down or destabilise the detection everything else depends on.
* A crop of one person is the top-down pose setting these models are best at.
  Run on a whole 1080p frame, a person forty metres down a fence line is a
  handful of pixels and their wrists are noise; scaled up from their own box,
  they are a person again.
* Keypoints land on a *track*, not on a fresh box, so a gesture can be voted on
  across frames (gesture.py) rather than believed the first time it is seen.

Returns keypoints in MODEL space. The caller maps them back with the same
``FrameTransform`` the detector uses — including its ``crop_x``/``crop_y``,
which exist for precisely this "tile within the original frame" case — so the
invariant holds here too: nothing downstream ever sees model-space coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..gesture import COCO_KEYPOINTS, Keypoint, Pose
from . import DetectorConfig, resolve_execution_providers
from .onnx_yolo import _preload_gpu_libraries, preprocess_into

logger = logging.getLogger(__name__)

__all__ = ["MockPoseEstimator", "OnnxPoseEstimator", "decode_pose_output"]

#: 4 box + 1 person score + 17 keypoints x (x, y, conf).
_EXPECTED_CHANNELS = 5 + len(COCO_KEYPOINTS) * 3


def decode_pose_output(raw: np.ndarray, min_person_conf: float) -> Pose | None:
    """Pick the best person in a YOLO-pose output and return their keypoints.

    Shape is ``[1, 56, N]`` — the same channels-first layout YOLO11 detection
    uses, so the transpose story is identical. Taking the single highest-scoring
    candidate rather than running NMS is not a shortcut: the input is a crop of
    one tracked person, so "which person is this" was already answered upstream
    by the tracker. Any second candidate in this crop is a bystander clipped by
    the box, and adopting their arms as the subject's would be worse than
    returning nothing.

    Pure and array-only so it can be tested against hand-built tensors without
    a weights file (tests/test_pose.py).
    """
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"expected a [1, C, N] pose output, got shape {raw.shape}")
    if raw.shape[1] != _EXPECTED_CHANNELS:
        raise ValueError(
            f"expected {_EXPECTED_CHANNELS} channels for {len(COCO_KEYPOINTS)} COCO "
            f"keypoints, got {raw.shape[1]}. This export is not a YOLO pose model."
        )

    preds = raw[0]  # [C, N]
    scores = preds[4]
    best = int(np.argmax(scores))
    if float(scores[best]) < min_person_conf:
        return None

    flat = preds[5:, best].reshape(len(COCO_KEYPOINTS), 3)
    return Pose(
        tuple(
            # Explicit float(): numpy scalars reaching the evidence canonicaliser
            # is a bug this project has already paid for once (README, "evidence
            # hashes were silently wrong"). Cast at the boundary, every time.
            Keypoint(float(x), float(y), float(conf))
            for x, y, conf in flat
        )
    )


class OnnxPoseEstimator:
    """One extra ONNX session, loaded only when gestures are enabled."""

    def __init__(self, cfg: DetectorConfig, min_person_conf: float = 0.40) -> None:
        self.cfg = cfg
        self.min_person_conf = min_person_conf
        weights = Path(cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"pose weights not found: {weights}\n"
                f"Run `make models` to download them (free, ~11 MB). "
                f"See docs/MODELS.md."
            )

        import onnxruntime as ort

        _preload_gpu_libraries(ort, cfg)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if cfg.intra_op_threads > 0:
            options.intra_op_num_threads = cfg.intra_op_threads

        providers = resolve_execution_providers(cfg, ort.get_available_providers())
        try:
            self._session = ort.InferenceSession(str(weights), options, providers=providers)
        except Exception:
            # Same P8 rule as the detector: never lose the pipeline to a broken
            # accelerator. Gestures on CPU are slower, not absent.
            logger.exception(
                "could not create a pose session with providers %s; retrying on CPU",
                [p[0] if isinstance(p, tuple) else p for p in providers],
            )
            self._session = ort.InferenceSession(
                str(weights), options, providers=["CPUExecutionProvider"]
            )

        self._input_name = self._session.get_inputs()[0].name
        self._buffer: np.ndarray | None = None
        logger.info(
            "onnx pose estimator loaded weights=%s providers=%s",
            weights,
            self._session.get_providers(),
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self.cfg.input_size

    def estimate(self, letterboxed: np.ndarray) -> Pose | None:
        """Keypoints for one already-letterboxed crop, in MODEL space.

        The caller does the letterboxing so that it owns the ``FrameTransform``
        that maps the answer back — the same division of labour the detector
        has, and the reason neither backend can silently return coordinates in
        a space the caller did not expect.
        """
        w, h = self.cfg.input_size
        if letterboxed.shape[:2] != (h, w):
            raise ValueError(
                f"pose input must be pre-letterboxed to {w}x{h}, got "
                f"{letterboxed.shape[1]}x{letterboxed.shape[0]}"
            )
        if self._buffer is None:
            self._buffer = np.empty((1, 3, h, w), dtype=np.float32)
        preprocess_into(letterboxed, self._buffer[0])
        outputs = self._session.run(None, {self._input_name: self._buffer})
        return decode_pose_output(np.asarray(outputs[0]), self.min_person_conf)


class MockPoseEstimator:
    """Scripted poses, no weights, no GPU. "Mock the GPU, not the logic."

    Hands back the queued poses in order and then keeps returning the last one,
    so a test can hold a gesture for as many frames as the voter needs without
    writing the same line eight times.
    """

    def __init__(self, poses: list[Pose | None] | None = None) -> None:
        self.poses = list(poses or [])
        self.calls = 0
        self.input_size = (640, 640)

    def estimate(self, letterboxed: np.ndarray) -> Pose | None:
        self.calls += 1
        if not self.poses:
            return None
        index = min(self.calls - 1, len(self.poses) - 1)
        return self.poses[index]
