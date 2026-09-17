"""Object detection backends (BUILD_SPEC §7.4). **[DEMO-CRITICAL]**

One ``Detector`` protocol, three backends:

* ``onnx``     — ONNX Runtime. CPU, CUDA, TensorRT or CoreML execution provider.
                 This is the default and the one `make demo` uses.
* ``tensorrt`` — a prebuilt TensorRT engine for the ``bop`` profile.
* ``mock``     — scripted, deterministic, no weights, no GPU. Tests use this.
                 "Mock the GPU, not the logic" (CLAUDE.md/Testing).

Blocker #3, cold start: ``build_detector`` warms up before returning, and the
worker does not report ready until it has. First TensorRT inference is ~2 s and
first ORT inference ~400 ms; without a warmup the demo looks frozen at exactly
the wrong moment.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..types import RawDetection

logger = logging.getLogger(__name__)

__all__ = [
    "Detector",
    "DetectorConfig",
    "MockDetector",
    "build_detector",
    "resolve_execution_providers",
]

# Tried in order; the first available one wins. CoreML keeps the macOS laptop
# profile usable, CUDA/TensorRT take over on a real box.
DEFAULT_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)

#: Providers for which ``fp16`` and the GPU tuning knobs mean anything. Setting
#: them on CPU or CoreML is not an error, it is just noise in the session log.
GPU_PROVIDERS: frozenset[str] = frozenset({"TensorrtExecutionProvider", "CUDAExecutionProvider"})


def _as_provider_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a config value into a tuple of provider names.

    ``tuple("CPUExecutionProvider")`` explodes a bare string into one tuple
    element per CHARACTER -- a classic Python footgun, and a real one here:
    a single-provider override (e.g. ``DRISHTI__DETECTOR__PROVIDERS=CPU...``)
    is exactly the kind of value an operator would type when working around a
    flaky GPU backend, and it silently produced a provider list of individual
    letters instead of failing loudly.
    """
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    backend: str = "onnx"
    model: str = "yolo11n"
    weights: str = "models/detect/yolo11n.onnx"
    input_size: tuple[int, int] = (640, 640)
    warmup_iterations: int = 10
    warmup_required: bool = True
    providers: tuple[str, ...] = DEFAULT_PROVIDERS
    intra_op_threads: int = 0
    fp16: bool = True
    # GPU knobs (§5, `bop` profile). Inert on CPU and CoreML.
    device_id: int = 0
    gpu_mem_limit_mb: int = 0  # 0 = let the driver decide
    cudnn_conv_algo_search: str = "HEURISTIC"
    trt_workspace_mb: int = 1024
    trt_engine_cache_dir: str = "models/trt_cache"
    trt_timing_cache: bool = True
    max_batch: int = 8
    max_wait_ms: int = 15
    nms_iou: float = 0.45
    max_detections: int = 100
    class_agnostic_nms: bool = True
    conf_thresholds: Mapping[str, float] = field(default_factory=lambda: {"default": 0.50})
    min_box_area_px: float = 200.0
    min_box_height_px: float = 18.0
    class_map: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> DetectorConfig:
        block = dict(cfg.get("detector", cfg))
        batching = dict(block.get("batching", {}))
        nms = dict(block.get("nms", {}))
        gpu = dict(block.get("gpu", {}))
        size = list(block.get("input_size", (640, 640)))
        raw_map = dict(block.get("class_map", {}))
        return cls(
            backend=str(block.get("backend", "onnx")),
            model=str(block.get("model", "yolo11n")),
            weights=str(block.get("weights", "models/detect/yolo11n.onnx")),
            input_size=(int(size[0]), int(size[1])),
            warmup_iterations=int(block.get("warmup_iterations", 10)),
            warmup_required=bool(block.get("warmup_required", True)),
            providers=_as_provider_tuple(block.get("providers", DEFAULT_PROVIDERS)),
            intra_op_threads=int(block.get("intra_op_threads", 0)),
            fp16=bool(block.get("fp16", True)),
            device_id=int(gpu.get("device_id", 0)),
            gpu_mem_limit_mb=int(gpu.get("gpu_mem_limit_mb", 0)),
            cudnn_conv_algo_search=str(gpu.get("cudnn_conv_algo_search", "HEURISTIC")).upper(),
            trt_workspace_mb=int(gpu.get("trt_workspace_mb", 1024)),
            trt_engine_cache_dir=str(gpu.get("trt_engine_cache_dir", "models/trt_cache")),
            trt_timing_cache=bool(gpu.get("trt_timing_cache", True)),
            max_batch=int(batching.get("max_batch", 8)),
            max_wait_ms=int(batching.get("max_wait_ms", 15)),
            nms_iou=float(nms.get("iou", 0.45)),
            max_detections=int(nms.get("max_detections", 100)),
            class_agnostic_nms=bool(nms.get("class_agnostic", True)),
            conf_thresholds=dict(block.get("conf_thresholds", {"default": 0.50})),
            min_box_area_px=float(block.get("min_box_area_px", 200.0)),
            min_box_height_px=float(block.get("min_box_height_px", 18.0)),
            class_map={int(k): dict(v) for k, v in raw_map.items()},
        )

    def threshold_for(self, cls: str) -> float:
        return float(self.conf_thresholds.get(cls, self.conf_thresholds.get("default", 0.50)))


def resolve_execution_providers(
    cfg: DetectorConfig, available: Sequence[str]
) -> list[str | tuple[str, dict[str, Any]]]:
    """Intersect the configured providers with what this ORT build offers, and
    attach each one's options. Pure: no session, no GPU, no import of ORT.

    Two things here are not cosmetic.

    **fp16 was previously read from config and then thrown away.** ORT does not
    infer precision from anywhere — an FP32 ONNX graph runs in FP32 on a GPU
    unless the execution provider is told otherwise, so ``fp16: true`` in the
    `bop` profile bought nothing until it was passed through as
    ``trt_fp16_enable``.

    **The TensorRT engine cache is what makes blocker #3 survivable.** TensorRT
    builds a kernel-tuned engine for the exact graph, shapes and device on the
    first inference, which takes tens of seconds to minutes — far past the ~2 s
    the warmup budget assumes. Cached, the second start deserialises in
    well under a second. Without the cache, every worker restart at a BOP pays
    the full build, which is not a thing anyone will wait for at 3 a.m.

    ``cudnn_conv_algo_search`` defaults to HEURISTIC rather than ORT's own
    EXHAUSTIVE default for the same reason: EXHAUSTIVE benchmarks every
    convolution algorithm at session init and adds seconds to startup for
    single-digit-percent steady-state gain.
    """
    offered = set(available)
    chosen = [p for p in cfg.providers if p in offered]
    if not chosen:
        logger.warning(
            "none of the configured execution providers %s are available in this "
            "onnxruntime build (%s); falling back to CPU. Expect reduced throughput.",
            list(cfg.providers),
            sorted(offered),
        )
        chosen = ["CPUExecutionProvider"]

    resolved: list[str | tuple[str, dict[str, Any]]] = []
    for name in chosen:
        if name == "TensorrtExecutionProvider":
            options: dict[str, Any] = {
                "device_id": cfg.device_id,
                "trt_fp16_enable": cfg.fp16,
                "trt_max_workspace_size": cfg.trt_workspace_mb * 1024 * 1024,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cfg.trt_engine_cache_dir,
                "trt_timing_cache_enable": cfg.trt_timing_cache,
            }
            resolved.append((name, options))
        elif name == "CUDAExecutionProvider":
            options = {
                "device_id": cfg.device_id,
                "cudnn_conv_algo_search": cfg.cudnn_conv_algo_search,
                "do_copy_in_default_stream": True,
            }
            if cfg.gpu_mem_limit_mb > 0:
                options["gpu_mem_limit"] = cfg.gpu_mem_limit_mb * 1024 * 1024
            resolved.append((name, options))
        else:
            resolved.append(name)
    return resolved


@runtime_checkable
class Detector(Protocol):
    """Frozen interface (§7.4). Boxes come back in MODEL space."""

    def infer(self, images: Sequence[Any]) -> list[list[RawDetection]]: ...

    def warmup(self, n: int = 10) -> float: ...

    @property
    def input_size(self) -> tuple[int, int]: ...

    @property
    def backend(self) -> str: ...


class MockDetector:
    """Deterministic detector for tests and for `make demo` without weights.

    Emits a person walking left-to-right and, optionally, a vehicle. Scripted,
    reproducible, and it never touches a GPU — which is the point. We mock the
    accelerator, never the tracking or rule logic under test.
    """

    def __init__(self, cfg: DetectorConfig | None = None, script: Any = None) -> None:
        self.cfg = cfg or DetectorConfig(backend="mock")
        self._script = script
        self._calls = 0

    def infer(self, images: Sequence[Any]) -> list[list[RawDetection]]:
        out: list[list[RawDetection]] = []
        for _ in images:
            if self._script is not None:
                out.append(list(self._script(self._calls)))
            else:
                x = 60 + (self._calls * 9) % 500
                out.append(
                    [
                        RawDetection(cls_id=0, conf=0.88, box=(x, 220.0, x + 46.0, 350.0)),
                        RawDetection(cls_id=2, conf=0.79, box=(400.0, 300.0, 560.0, 400.0)),
                    ]
                )
            self._calls += 1
        return out

    def warmup(self, n: int = 10) -> float:
        return 0.0

    @property
    def input_size(self) -> tuple[int, int]:
        return self.cfg.input_size

    @property
    def backend(self) -> str:
        return "mock"


def build_detector(cfg: DetectorConfig) -> Detector:
    """Factory. Reads config only — no side effects beyond loading and warming.

    Warms up before returning (blocker #3). A warmup failure with
    ``warmup_required`` raises: better a clear startup error than a demo that
    appears to hang on its first frame.
    """
    backend = cfg.backend.lower()
    detector: Detector

    if backend == "mock":
        detector = MockDetector(cfg)
    elif backend in ("onnx", "tensorrt"):
        from .onnx_yolo import OnnxYoloDetector

        detector = OnnxYoloDetector(cfg)
    else:
        raise ValueError(
            f"unknown detector backend {cfg.backend!r}; expected onnx, tensorrt or mock"
        )

    if cfg.warmup_iterations > 0:
        started = time.monotonic()
        try:
            spent = detector.warmup(cfg.warmup_iterations)
        except Exception as exc:
            if cfg.warmup_required:
                raise RuntimeError(
                    f"detector warmup failed for backend={cfg.backend} "
                    f"weights={cfg.weights}: {exc}"
                ) from exc
            logger.exception("detector warmup failed; continuing because warmup_required=false")
            spent = time.monotonic() - started
        logger.info(
            "detector ready backend=%s model=%s input=%dx%d warmup=%d iters in %.2fs",
            detector.backend,
            cfg.model,
            cfg.input_size[0],
            cfg.input_size[1],
            cfg.warmup_iterations,
            spent,
        )
    return detector
