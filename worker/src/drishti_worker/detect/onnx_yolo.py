"""ONNX Runtime YOLO detector (BUILD_SPEC §7.4). **hot path**

Handles YOLO11/YOLOv8 ONNX exports (output ``[1, 4+nc, N]``) and RT-DETR
(``[1, N, 4+nc]``), because switching to RT-DETR is our Apache-2.0 escape hatch
from YOLO's AGPL licence and it must be a config change, not a code change
(docs/MODELS.md §2).

Returns boxes in MODEL space. ``pipeline.py`` maps them back through the
``FrameTransform`` immediately, and nothing downstream ever sees model-space
coordinates (invariant).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..types import RawDetection
from . import DetectorConfig

logger = logging.getLogger(__name__)

__all__ = ["OnnxYoloDetector"]


class OnnxYoloDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        weights = Path(cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"detector weights not found: {weights}\n"
                f"Run `make models` to download them (free, ~166 MB total). "
                f"See docs/MODELS.md."
            )

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if cfg.intra_op_threads > 0:
            options.intra_op_num_threads = cfg.intra_op_threads

        available = set(ort.get_available_providers())
        providers = [p for p in cfg.providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
            logger.warning(
                "none of the configured execution providers %s are available; "
                "falling back to CPU. Expect reduced throughput.",
                list(cfg.providers),
            )

        self._session = ort.InferenceSession(str(weights), options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._providers = self._session.get_providers()
        logger.info(
            "onnx detector loaded weights=%s providers=%s", weights, self._providers
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self.cfg.input_size

    @property
    def backend(self) -> str:
        return f"onnx[{self._providers[0]}]" if self._providers else "onnx"

    def warmup(self, n: int = 10) -> float:
        """Blocker #3. Run dummy inferences so the first real frame is not slow.

        ORT builds and optimises its graph, and TensorRT builds or deserialises
        an engine, on the FIRST call. Doing that while an evaluator watches an
        empty dashboard is how a working system looks broken.
        """
        w, h = self.cfg.input_size
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        started = time.monotonic()
        for i in range(max(1, n)):
            self._session.run(None, {self._input_name: dummy})
            if i == 0:
                logger.info(
                    "detector first inference took %.0f ms (cold start)",
                    (time.monotonic() - started) * 1000,
                )
        return time.monotonic() - started

    def infer(self, images: Sequence[Any]) -> list[list[RawDetection]]:
        if not images:
            return []
        batch = np.stack([self._preprocess(img) for img in images])
        outputs = self._session.run(None, {self._input_name: batch})[0]
        return [self._postprocess(outputs[i]) for i in range(len(images))]

    # -- internals ---------------------------------------------------------

    def _preprocess(self, image: Any) -> np.ndarray:
        """BGR uint8 HWC -> RGB float32 CHW, letterboxed.

        The caller has already resized to ``input_size`` and holds the matching
        ``FrameTransform``; doing the resize twice would desynchronise the
        transform from the pixels.
        """
        rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))

    def _postprocess(self, output: np.ndarray) -> list[RawDetection]:
        # YOLO11/v8: (4+nc, N). RT-DETR: (N, 4+nc). Distinguish by which axis
        # looks like a prediction count.
        preds = output.T if output.shape[0] < output.shape[1] else output
        if preds.shape[1] < 5:
            raise ValueError(f"unexpected detector output shape {output.shape}")

        boxes_cxcywh = preds[:, :4]
        scores_all = preds[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        confs = scores_all.max(axis=1)

        lowest = (
            min(self.cfg.conf_thresholds.values()) if self.cfg.conf_thresholds else 0.25
        )
        keep = confs >= lowest
        if not keep.any():
            return []

        boxes_cxcywh = boxes_cxcywh[keep]
        cls_ids = cls_ids[keep]
        confs = confs[keep]

        cx, cy, bw, bh = boxes_cxcywh.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        selected = self._nms(xyxy, confs, cls_ids)
        return [
            RawDetection(
                cls_id=int(cls_ids[i]),
                conf=float(confs[i]),
                box=(
                    float(xyxy[i, 0]),
                    float(xyxy[i, 1]),
                    float(xyxy[i, 2]),
                    float(xyxy[i, 3]),
                ),
            )
            for i in selected
        ]

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray
    ) -> list[int]:
        """Greedy NMS, vectorised. Class-agnostic by default.

        Class-agnostic matters here: YOLO cheerfully reports the same vehicle as
        both 'car' and 'truck', and two boxes on one object become two tracks,
        two rule firings and two alerts.
        """
        if boxes.size == 0:
            return []
        order = scores.argsort()[::-1]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep: list[int] = []

        while order.size > 0 and len(keep) < self.cfg.max_detections:
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
            ious = inter / np.maximum(1e-9, areas[i] + areas[rest] - inter)
            mask = ious <= self.cfg.nms_iou
            if not self.cfg.class_agnostic_nms:
                mask |= cls_ids[rest] != cls_ids[i]
            order = rest[mask]
        return keep
