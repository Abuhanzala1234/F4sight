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
from . import (
    GPU_PROVIDERS,
    DetectorConfig,
    first_provider_is_gpu,
    gpu_run_lock,
    resolve_execution_providers,
    session_options,
)

logger = logging.getLogger(__name__)

__all__ = ["OnnxYoloDetector", "preprocess_into"]

#: Divide, do not multiply by a precomputed 1/255. The reciprocal is not
#: representable in binary floating point, so scaling by it moves about half of
#: all pixel values by one ULP away from what a plain division gives. The error
#: is far below anything a detector could notice, but it would stop this rewrite
#: from being verifiably identical to the implementation it replaced — and a
#: hot-path rewrite that is only approximately equivalent is one nobody can
#: check. The division is memory-bandwidth bound here anyway; it measured the
#: same as the multiply.
_SCALE = np.float32(255.0)


def preprocess_into(image: np.ndarray, dst: np.ndarray) -> None:
    """BGR uint8 HWC -> RGB float32 CHW, written into ``dst``. **hot path**

    Module level rather than a method so the arithmetic can be tested against
    the obvious reference implementation without a weights file or a GPU.

    ``transpose(2, 0, 1)`` is a free view; ``[::-1]`` on the resulting channel
    axis is the BGR->RGB swap, also free. ``copyto`` does the strided read and
    the uint8->float32 cast in a single pass, then the scale happens in place.
    No temporaries: see :meth:`OnnxYoloDetector._prepare_batch` for why that
    matters here.
    """
    np.copyto(dst, image.transpose(2, 0, 1)[::-1], casting="unsafe")
    dst /= _SCALE


def _preload_gpu_libraries(ort: Any, cfg: DetectorConfig) -> None:
    """Make the pip-installed CUDA/cuDNN libraries findable before ORT looks.

    Worth the twelve lines, because the failure it prevents is deeply
    unhelpful. The CUDA libraries ship as ``nvidia-*`` wheels that unpack to
    ``site-packages/nvidia/**/bin``, which is not on the Windows DLL search
    path. Without this call ORT cannot load ``onnxruntime_providers_cuda.dll``,
    reports the *dependency* as missing (``cublasLt64_13.dll``), and quietly
    runs on CPU — a working demo that is ten times slower than the box it is
    running on, with no obvious cause. Observed on exactly this hardware.

    ``preload_dlls`` arrived in onnxruntime 1.21 and is a no-op where the
    libraries come from a system CUDA install instead. Failing to preload is
    never fatal: the session creation below falls back to CPU and says so.
    """
    if not any(p in GPU_PROVIDERS for p in cfg.providers):
        return
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        logger.debug(
            "onnxruntime %s has no preload_dlls; relying on the system CUDA "
            "libraries being on PATH",
            getattr(ort, "__version__", "?"),
        )
        return
    try:
        preload()
    except Exception:
        logger.exception(
            "onnxruntime.preload_dlls() failed; if the GPU provider does not "
            "bind below, the CUDA/cuDNN libraries are not discoverable"
        )


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

        _preload_gpu_libraries(ort, cfg)

        providers = resolve_execution_providers(cfg, ort.get_available_providers())
        options = session_options(ort, cfg, gpu=first_provider_is_gpu(providers))
        if any(
            (p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider" for p in providers
        ):
            # ORT will not create the cache directory itself; a missing path
            # silently disables caching and every start rebuilds the engine.
            Path(cfg.trt_engine_cache_dir).mkdir(parents=True, exist_ok=True)

        try:
            self._session = ort.InferenceSession(str(weights), options, providers=providers)
        except Exception:
            # A GPU provider can be present in the build but unusable on the
            # box — missing CUDA/cuDNN/TensorRT libraries, a driver too old, or
            # another process holding the VRAM. P8 says never lose the pipeline
            # to a broken accelerator: log loudly, then run on CPU.
            logger.exception(
                "could not create an inference session with providers %s; "
                "retrying on CPU. Throughput will drop — check the CUDA/TensorRT "
                "runtime libraries on this host.",
                [p[0] if isinstance(p, tuple) else p for p in providers],
            )
            self._session = ort.InferenceSession(
                str(weights),
                session_options(ort, cfg, gpu=False),
                providers=["CPUExecutionProvider"],
            )

        self._input_name = self._session.get_inputs()[0].name
        self._providers = self._session.get_providers()
        self._buffer: np.ndarray | None = None
        self._run_lock = gpu_run_lock(self._session)
        #: Cumulative time per phase of infer(); the stats loop logs deltas,
        #: so a slow detector says WHERE it is slow, not just that it is.
        self.phase_ms: dict[str, float] = dict.fromkeys(
            ("prepare", "gpu_queue_wait", "run", "postprocess"), 0.0
        )
        logger.info(
            "onnx detector loaded weights=%s providers=%s fp16=%s",
            weights,
            self._providers,
            cfg.fp16 if self.on_gpu else "n/a (cpu)",
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self.cfg.input_size

    @property
    def backend(self) -> str:
        return f"onnx[{self._providers[0]}]" if self._providers else "onnx"

    @property
    def providers(self) -> list[str]:
        """The providers ORT actually bound, in priority order.

        Reported in health and in `make bench` output: "we asked for TensorRT"
        and "TensorRT is running" are different claims, and only the second one
        belongs in a benchmark table.
        """
        return list(self._providers)

    @property
    def on_gpu(self) -> bool:
        return bool(self._providers) and self._providers[0] in GPU_PROVIDERS

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
            with self._run_lock:
                self._session.run(None, {self._input_name: dummy})
            if i == 0:
                logger.info(
                    "detector first inference took %.0f ms (cold start)",
                    (time.monotonic() - started) * 1000,
                )
        # On a GPU every new batch shape is planned on first use, and live
        # batches are 1..max_batch -- warming only size 1 left the first real
        # multi-camera batches to pay that cost (~470ms, measured) on air.
        if self.on_gpu:
            for size in range(2, max(2, self.cfg.max_batch) + 1):
                batch = np.zeros((size, 3, h, w), dtype=np.float32)
                for _ in range(2):
                    with self._run_lock:
                        self._session.run(None, {self._input_name: batch})
        return time.monotonic() - started

    def infer(self, images: Sequence[Any]) -> list[list[RawDetection]]:
        if not images:
            return []
        t0 = time.perf_counter()
        batch = self._prepare_batch(images)
        t1 = time.perf_counter()
        with self._run_lock:
            t2 = time.perf_counter()
            outputs = self._session.run(None, {self._input_name: batch})[0]
        t3 = time.perf_counter()
        results = [self._postprocess(outputs[i]) for i in range(len(images))]
        t = self.phase_ms
        t["prepare"] += (t1 - t0) * 1000
        t["gpu_queue_wait"] += (t2 - t1) * 1000
        t["run"] += (t3 - t2) * 1000
        t["postprocess"] += (time.perf_counter() - t3) * 1000
        return results

    # -- internals ---------------------------------------------------------

    def _prepare_batch(self, images: Sequence[Any]) -> np.ndarray:
        """Fill a reused NCHW buffer with the batch. **hot path**

        Measured on an RTX 3050 with yolo11n at 640x640: preprocessing cost
        5.8 ms per frame against 5.4 ms of actual GPU inference — the data
        preparation had become the more expensive half of "inference". Two
        causes, both fixed here:

        * **Temporaries.** ``image[:, :, ::-1].astype(float32) / 255`` walks a
          4.9 MB array three times and allocates two full-size intermediates
          per frame. Converting into a destination buffer walks it twice with
          none.
        * **Re-allocation.** ``np.stack`` allocated a fresh 39 MB array for
          every batch of 8, at analytics frame rate, forever.

        Safe to reuse the buffer because §3.2 gives the pipeline exactly one
        shared inference thread; ORT has copied the data to the device by the
        time ``run`` returns. If a second thread ever calls ``infer``
        concurrently, this buffer needs a lock or a per-thread copy.
        """
        count = len(images)
        width, height = self.cfg.input_size
        buffer = self._buffer
        if buffer is None or buffer.shape[0] < count:
            buffer = np.empty((count, 3, height, width), dtype=np.float32)
            self._buffer = buffer

        view = buffer[:count]  # contiguous: slicing only the leading axis
        for i, image in enumerate(images):
            preprocess_into(image, view[i])
        return view

    def _preprocess(self, image: Any) -> np.ndarray:
        """Single-frame form of :meth:`_preprocess_into`, for callers outside
        the batch path (benchmarks, tests). Allocates; the hot path does not.
        """
        width, height = self.cfg.input_size
        out = np.empty((3, height, width), dtype=np.float32)
        preprocess_into(image, out)
        return out

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

        lowest = min(self.cfg.conf_thresholds.values()) if self.cfg.conf_thresholds else 0.25
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

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> list[int]:
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
