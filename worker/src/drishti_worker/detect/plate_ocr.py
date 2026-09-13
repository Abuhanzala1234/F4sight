"""Plate detection + OCR backend (BUILD_SPEC §7.9).

Two free models, both ONNX, both local: a YOLO11n fine-tune that finds the plate
inside a vehicle crop, and PaddleOCR v4's English recogniser (Apache-2.0) that
reads it. When the recogniser weights are absent we fall back to Tesseract with
a plate charset, which is worse but free and already on most systems.

All the *judgement* — validation, confusion correction, multi-frame voting —
lives in ``anpr.py`` as pure functions. This module only produces candidate
strings. That split is what lets the interesting logic be property-tested
without any model on disk.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..anpr import PlateCandidate

logger = logging.getLogger(__name__)

__all__ = ["OnnxPlateReader"]

PLATE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class OnnxPlateReader:
    """Localise a plate in a vehicle crop and read it."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        block = dict(cfg.get("anpr", cfg))
        self.min_plate_width = int(block.get("min_plate_width_px", 80))
        self.plate_conf = float(block.get("plate_conf", 0.45))
        self._det_path = Path(block.get("plate_detector_weights", ""))
        self._rec_path = Path(block.get("ocr_rec_weights", ""))
        self._det: Any = None
        self._rec: Any = None
        self._use_tesseract = not self._rec_path.exists()
        if self._use_tesseract:
            logger.warning(
                "plate recogniser weights missing at %s; falling back to Tesseract. "
                "Run `make models` for the better free option.",
                self._rec_path,
            )

    def read(self, crop: Any, frame_id: int = 0) -> PlateCandidate | None:
        """Return one raw candidate from one vehicle crop, or None."""
        plate = self._localise(crop)
        if plate is None:
            return None
        box, region = plate
        if region.shape[1] < self.min_plate_width:
            # Below this width OCR is guessing, and P3 says do not guess.
            return None
        text, conf, char_confs = self._recognise(region)
        if not text:
            return None
        return PlateCandidate(
            raw_text=text,
            conf=conf,
            box=box,
            char_confs=tuple(char_confs),
            frame_id=frame_id,
        )

    # -- internals ---------------------------------------------------------

    def _localise(
        self, crop: Any
    ) -> tuple[tuple[float, float, float, float], Any] | None:
        import numpy as np

        if self._det is None:
            if not self._det_path.exists():
                # No plate detector: assume the lower third of the vehicle,
                # which is where plates are, and let voting reject the noise.
                h, w = crop.shape[:2]
                y1 = int(h * 0.60)
                return (0.0, float(y1), float(w), float(h)), crop[y1:, :]
            import onnxruntime as ort

            self._det = ort.InferenceSession(
                str(self._det_path), providers=["CPUExecutionProvider"]
            )

        import cv2

        h, w = crop.shape[:2]
        inp = cv2.resize(crop, (320, 320)).astype(np.float32)[:, :, ::-1] / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]
        out = self._det.run(None, {self._det.get_inputs()[0].name: inp})[0]
        preds = out[0].T if out[0].shape[0] < out[0].shape[1] else out[0]
        if preds.size == 0:
            return None
        scores = preds[:, 4:].max(axis=1)
        best = int(scores.argmax())
        if float(scores[best]) < self.plate_conf:
            return None
        cx, cy, bw, bh = preds[best, :4]
        sx, sy = w / 320.0, h / 320.0
        x1 = max(0, int((cx - bw / 2) * sx))
        y1 = max(0, int((cy - bh / 2) * sy))
        x2 = min(w, int((cx + bw / 2) * sx))
        y2 = min(h, int((cy + bh / 2) * sy))
        if x2 <= x1 or y2 <= y1:
            return None
        return (float(x1), float(y1), float(x2), float(y2)), crop[y1:y2, x1:x2]

    def _recognise(self, region: Any) -> tuple[str, float, list[float]]:
        if self._use_tesseract:
            return self._tesseract(region)
        try:
            return self._paddle(region)
        except Exception:
            logger.exception("PaddleOCR recognition failed; falling back to Tesseract")
            return self._tesseract(region)

    def _tesseract(self, region: Any) -> tuple[str, float, list[float]]:
        try:
            import pytesseract
        except ImportError:
            logger.error(
                "neither the ONNX recogniser nor pytesseract is available; "
                "ANPR cannot read plates. Run `make models` or `pip install pytesseract`."
            )
            return "", 0.0, []

        import cv2

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config = f"--psm 7 -c tessedit_char_whitelist={PLATE_CHARSET}"
        data = pytesseract.image_to_data(
            gray, config=config, output_type=pytesseract.Output.DICT
        )
        words = [w for w in data["text"] if w.strip()]
        confs = [float(c) / 100.0 for c in data["conf"] if float(c) >= 0]
        if not words:
            return "", 0.0, []
        text = "".join(words)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return text, mean_conf, confs

    def _paddle(self, region: Any) -> tuple[str, float, list[float]]:
        import cv2
        import numpy as np

        if self._rec is None:
            import onnxruntime as ort

            self._rec = ort.InferenceSession(
                str(self._rec_path), providers=["CPUExecutionProvider"]
            )
        img = cv2.resize(region, (320, 48)).astype(np.float32)
        img = (img[:, :, ::-1] / 255.0 - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[None]
        logits = self._rec.run(None, {self._rec.get_inputs()[0].name: img})[0][0]

        # CTC greedy decode with blank at index 0.
        indices = logits.argmax(axis=1)
        probs = logits.max(axis=1)
        chars: list[str] = []
        confs: list[float] = []
        previous = -1
        for idx, prob in zip(indices, probs, strict=True):
            if idx != previous and idx > 0:
                pos = int(idx) - 1
                if pos < len(PLATE_CHARSET):
                    chars.append(PLATE_CHARSET[pos])
                    confs.append(float(prob))
            previous = int(idx)
        if not chars:
            return "", 0.0, []
        return "".join(chars), sum(confs) / len(confs), confs
