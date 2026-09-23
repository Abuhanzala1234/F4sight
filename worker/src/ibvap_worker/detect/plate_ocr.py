"""Plate detection + OCR backend (BUILD_SPEC §7.9).

Two free models, both ONNX, both local: a YOLO11n fine-tune that finds the plate
inside a vehicle crop, and PaddleOCR v4's English recogniser (Apache-2.0) that
reads it. Neither ships with this checkout -- `models/anpr/` does not exist and
nothing in `scripts/fetch_models.py` fetches them, so on a fresh clone this
falls back automatically, in order:

1. **RapidOCR** (Apache-2.0, `pip install rapidocr-onnxruntime`) -- a pretrained
   general OCR engine that runs on the ``onnxruntime`` this project already
   depends on. No system package, no GPU, no training: it ships its own
   detection+recognition ONNX weights and reads arbitrary alphanumeric text
   out of the box, a plate crop included. This is what actually reads plates
   in this checkout right now.
2. **Tesseract** via ``pytesseract``, if RapidOCR is not installed either --
   needs the separate ``tesseract-ocr`` system package, so it is usually the
   one that is NOT available in a sandboxed/no-sudo environment.

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


def _build_rapidocr(threads: int, det_limit_side_len: int) -> Any:
    """RapidOCR tuned for plate strips, not documents (see config/anpr.yaml).

    RapidOCR builds its ORT sessions with a bare ``SessionOptions()`` and has
    no thread setting, so each of its sessions defaults to one thread per core
    and contends with the primary detector. The only hook is the name its
    ``utils`` module looks up at construction time, swapped for the duration
    of this one call and always restored.
    """
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime import utils as rapid_utils

    original = rapid_utils.SessionOptions

    def capped() -> Any:
        opts = original()
        opts.intra_op_num_threads = threads
        return opts

    rapid_utils.SessionOptions = capped
    try:
        return RapidOCR(
            use_angle_cls=False,
            det_model_path=None,
            det_limit_side_len=det_limit_side_len,
            det_limit_type="min",
        )
    finally:
        rapid_utils.SessionOptions = original


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
        self._rapidocr: Any = None
        self._use_tesseract = not self._rec_path.exists()
        # Availability is resolved ONCE here, not re-imported and re-logged
        # per crop inside _recognise() -- that used to re-trigger the exact
        # same ImportError on every vehicle track and drown the log within
        # seconds on a busy scene.
        self._rapidocr_available = False
        self._tesseract_available = False
        self._ocr_threads = int(block.get("ocr_threads", 2))
        self._ocr_det_limit = int(block.get("ocr_det_limit_side_len", 320))
        if self._use_tesseract:
            try:
                import rapidocr_onnxruntime  # noqa: F401
            except ImportError:
                pass
            else:
                self._rapidocr_available = True
                # Built here, at worker startup, not on the first vehicle:
                # loading three ONNX models lazily used to stall the live
                # per-camera thread mid-stream the first time a car appeared.
                self._rapidocr = _build_rapidocr(self._ocr_threads, self._ocr_det_limit)
                logger.warning(
                    "plate recogniser weights missing at %s; reading plates "
                    "with RapidOCR (pretrained, generic) instead of the "
                    "plate-tuned PaddleOCR model. Run `make models` for the "
                    "better, plate-specific option.",
                    self._rec_path,
                )
            if not self._rapidocr_available:
                try:
                    import pytesseract  # noqa: F401
                except ImportError:
                    logger.error(
                        "no plate OCR backend available: the ONNX recogniser, "
                        "RapidOCR and pytesseract are all missing. ANPR "
                        "cannot read plates. `pip install rapidocr-onnxruntime` "
                        "is the fastest fix -- no system package needed."
                    )
                else:
                    self._tesseract_available = True
                    logger.warning(
                        "plate recogniser weights missing at %s; falling back "
                        "to Tesseract. Run `make models` for the better free "
                        "option.",
                        self._rec_path,
                    )
        else:
            self._tesseract_available = True

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

    def _localise(self, crop: Any) -> tuple[tuple[float, float, float, float], Any] | None:
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
        if not self._use_tesseract:
            try:
                return self._paddle(region)
            except Exception:
                logger.exception("PaddleOCR recognition failed; falling back")
        if self._rapidocr_available:
            try:
                return self._rapidocr_read(region)
            except Exception:
                logger.exception("RapidOCR recognition failed; falling back to Tesseract")
        return self._tesseract(region)

    def _rapidocr_read(self, region: Any) -> tuple[str, float, list[float]]:
        if self._rapidocr is None:
            self._rapidocr = _build_rapidocr(self._ocr_threads, self._ocr_det_limit)
        result, _elapse = self._rapidocr(region)
        if not result:
            return "", 0.0, []

        # RapidOCR is a general text reader, not plate-specific -- a noisy
        # crop can return more than one line (a sticker, a bolt reflection).
        # Picking the reading with the most plate-charset characters survives
        # that better than picking whichever one happened to score highest
        # raw confidence, since junk text often scores confidently too.
        best_text, best_conf, best_score = "", 0.0, -1
        for _box, text, conf_str in result:
            cleaned = "".join(ch for ch in text.upper() if ch in PLATE_CHARSET)
            if len(cleaned) > best_score:
                best_text, best_conf, best_score = cleaned, float(conf_str), len(cleaned)
        if not best_text:
            return "", 0.0, []
        return best_text, best_conf, [best_conf] * len(best_text)

    def _tesseract(self, region: Any) -> tuple[str, float, list[float]]:
        # Checked once in __init__, not re-imported and re-logged here on every
        # crop -- see that check's comment for why this used to flood the log.
        if not self._tesseract_available:
            return "", 0.0, []
        import cv2
        import pytesseract

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config = f"--psm 7 -c tessedit_char_whitelist={PLATE_CHARSET}"
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
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
