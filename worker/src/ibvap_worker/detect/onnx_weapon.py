"""ONNX Runtime weapon detector (BUILD_SPEC §7.7, weapons).

A YOLOv8 two-class detector (``guns``, ``knife``) run over crops of person
tracks. Output layout is ``[1, 4+nc, N]`` — the same channels-first tensor
YOLO11 detection produces, so the transpose and NMS story is the one this
codebase already knows.

Returns boxes in MODEL space. The caller maps them back through the crop's
``FrameTransform``, keeping the §7.4 invariant that nothing downstream ever
sees model-space coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..weapon import WEAPON_CLASSES, WeaponCandidate
from . import DetectorConfig, resolve_execution_providers
from .onnx_yolo import _preload_gpu_libraries, preprocess_into

logger = logging.getLogger(__name__)

__all__ = ["MockWeaponDetector", "OnnxWeaponDetector", "decode_weapon_output", "nms"]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Indices of the boxes to keep.

    Module level and array-only so it can be tested against hand-built boxes
    without a weights file.
    """
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_threshold]
    return keep


def decode_weapon_output(
    raw: np.ndarray,
    min_conf: float,
    iou_threshold: float = 0.45,
    debug: dict[str, Any] | None = None,
) -> WeaponCandidate | None:
    """Strongest surviving weapon in one crop, or None.

    Only the best detection is returned. The crop is one tracked person and the
    question this answers is "is this person armed", not "inventory everything
    in frame" -- a second, weaker box in the same crop changes no decision
    downstream, and reporting it would only invite the operator to triage it.

    ``debug``, when given, is filled with the best score the model saw ACROSS
    EVERY ANCHOR, regardless of ``min_conf`` -- purely observational, and
    intentionally never returned as the function's actual result. Below
    ``min_conf`` this returns None either way; ``min_conf`` is a measured,
    documented safety floor (see weapons.yaml) and nothing here loosens it.
    Without this, a below-threshold sighting leaves no trace anywhere, and
    "the model saw nothing" and "the model saw something at 0.60" are
    indistinguishable from the logs -- which is exactly the question an
    operator asks the first time a real test does not alert.
    """
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"expected a [1, C, N] detector output, got shape {raw.shape}")
    expected = 4 + len(WEAPON_CLASSES)
    if raw.shape[1] != expected:
        raise ValueError(
            f"expected {expected} channels for classes {WEAPON_CLASSES}, got "
            f"{raw.shape[1]}. This export is not the weapon model this config expects."
        )

    preds = raw[0]
    class_scores = preds[4:]
    best_cls = class_scores.argmax(axis=0)
    best_score = class_scores.max(axis=0)

    if debug is not None and best_score.size:
        top = int(best_score.argmax())
        debug["best_conf"] = float(best_score[top])
        debug["best_cls"] = str(WEAPON_CLASSES[int(best_cls[top])])

    keep_mask = best_score >= min_conf
    if not keep_mask.any():
        return None

    cx, cy, bw, bh = (
        preds[0][keep_mask],
        preds[1][keep_mask],
        preds[2][keep_mask],
        preds[3][keep_mask],
    )
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    scores = best_score[keep_mask]
    classes = best_cls[keep_mask]

    survivors: list[tuple[float, int]] = []
    for cls_id in np.unique(classes):
        sel = classes == cls_id
        sub_boxes, sub_scores = boxes[sel], scores[sel]
        for k in nms(sub_boxes, sub_scores, iou_threshold):
            survivors.append((float(sub_scores[k]), int(cls_id)))
    if not survivors:
        return None

    conf, cls_id = max(survivors)
    # Explicit float()/str(): numpy scalars reaching the evidence canonicaliser
    # is a bug this project has already paid for once (README).
    return WeaponCandidate(cls=str(WEAPON_CLASSES[cls_id]), conf=float(conf))


class OnnxWeaponDetector:
    """One extra ONNX session, loaded only when weapon detection is enabled."""

    def __init__(self, cfg: DetectorConfig, min_conf: float = 0.55, nms_iou: float = 0.45) -> None:
        self.cfg = cfg
        self.min_conf = min_conf
        self.nms_iou = nms_iou
        weights = Path(cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"weapon weights not found: {weights}\n"
                f"Run `make models` to download them (free, ~12 MB). See docs/MODELS.md."
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
            # P8: never lose the pipeline to a broken accelerator.
            logger.exception(
                "could not create a weapon session with providers %s; retrying on CPU",
                [p[0] if isinstance(p, tuple) else p for p in providers],
            )
            self._session = ort.InferenceSession(
                str(weights), options, providers=["CPUExecutionProvider"]
            )

        self._input_name = self._session.get_inputs()[0].name
        self._buffer: np.ndarray | None = None
        logger.info(
            "onnx weapon detector loaded weights=%s providers=%s",
            weights,
            self._session.get_providers(),
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self.cfg.input_size

    def detect(self, letterboxed: np.ndarray) -> WeaponCandidate | None:
        """Best weapon in one already-letterboxed crop, or None."""
        w, h = self.cfg.input_size
        if letterboxed.shape[:2] != (h, w):
            raise ValueError(
                f"weapon input must be pre-letterboxed to {w}x{h}, got "
                f"{letterboxed.shape[1]}x{letterboxed.shape[0]}"
            )
        if self._buffer is None:
            self._buffer = np.empty((1, 3, h, w), dtype=np.float32)
        preprocess_into(letterboxed, self._buffer[0])
        outputs = self._session.run(None, {self._input_name: self._buffer})
        return decode_weapon_output(np.asarray(outputs[0]), self.min_conf, self.nms_iou)


class MockWeaponDetector:
    """Scripted results, no weights, no GPU. "Mock the GPU, not the logic.\" """

    def __init__(self, results: list[WeaponCandidate | None] | None = None) -> None:
        self.results = list(results or [])
        self.calls = 0
        self.input_size = (640, 640)

    def detect(self, letterboxed: np.ndarray) -> WeaponCandidate | None:
        self.calls += 1
        if not self.results:
            return None
        return self.results[min(self.calls - 1, len(self.results) - 1)]
