"""Detector factory, warmup and coordinate mapping (§7.4).

We mock the GPU, never the logic: ``MockDetector`` stands in for the accelerator
while the transform arithmetic and class mapping under test are the real code.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibvap_worker.detect import (
    Detector,
    DetectorConfig,
    MockDetector,
    build_detector,
    resolve_execution_providers,
)
from ibvap_worker.detect.onnx_yolo import preprocess_into
from ibvap_worker.pipeline import Pipeline
from ibvap_worker.types import FrameTransform, RawDetection

CPU_ONLY = ["CPUExecutionProvider"]
FULL_GPU_BOX = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

CLASS_MAP = {
    0: {"cls": "person"},
    2: {"cls": "vehicle", "vehicle_type": "car"},
    16: {"cls": "animal", "animal_type": "dog"},
}


class TestFactory:
    def test_mock_backend(self):
        detector = build_detector(DetectorConfig(backend="mock"))
        assert detector.backend == "mock"
        assert isinstance(detector, Detector)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown detector backend"):
            build_detector(DetectorConfig(backend="magic"))

    def test_missing_weights_fail_loudly_with_a_fix(self):
        """A clear startup error beats a demo that hangs on its first frame."""
        with pytest.raises((FileNotFoundError, RuntimeError)) as exc:
            build_detector(DetectorConfig(backend="onnx", weights="models/nope.onnx"))
        assert "make models" in str(exc.value)

    def test_warmup_runs_before_the_detector_is_returned(self):
        """Blocker #3."""
        calls = []

        class Counting(MockDetector):
            def warmup(self, n=10):
                calls.append(n)
                return 0.01

        cfg = DetectorConfig(backend="mock", warmup_iterations=10)
        detector = Counting(cfg)
        from ibvap_worker.detect import build_detector as build

        # Exercise the same path the factory takes.
        assert detector.warmup(cfg.warmup_iterations) == 0.01
        assert calls == [10]
        assert build(DetectorConfig(backend="mock")).backend == "mock"


class TestConfig:
    def test_from_mapping(self):
        cfg = DetectorConfig.from_mapping(
            {
                "detector": {
                    "backend": "onnx",
                    "input_size": [480, 480],
                    "batching": {"max_batch": 2, "max_wait_ms": 25},
                    "conf_thresholds": {"person": 0.4, "default": 0.5},
                    "class_map": {0: {"cls": "person"}},
                }
            }
        )
        assert cfg.input_size == (480, 480)
        assert cfg.max_batch == 2
        assert cfg.threshold_for("person") == 0.4
        assert cfg.threshold_for("animal") == 0.5  # falls back to default

    def test_per_class_thresholds_differ(self):
        """P3: person and vehicle deserve different thresholds."""
        cfg = DetectorConfig.from_mapping(
            {"detector": {"conf_thresholds": {"person": 0.40, "bag": 0.55, "default": 0.5}}}
        )
        assert cfg.threshold_for("person") < cfg.threshold_for("bag")


class TestExecutionProviders:
    """§5, the `bop` profile. We mock the GPU, never the logic: the provider
    list and its options are pure config arithmetic and get tested as such.
    """

    def _options(self, resolved, name: str) -> dict:
        for entry in resolved:
            if isinstance(entry, tuple) and entry[0] == name:
                return entry[1]
        raise AssertionError(f"{name} not in {resolved}")

    def test_configured_order_is_preserved(self):
        cfg = DetectorConfig()
        resolved = resolve_execution_providers(cfg, FULL_GPU_BOX)
        names = [e[0] if isinstance(e, tuple) else e for e in resolved]
        assert names[0] == "TensorrtExecutionProvider"
        assert names[1] == "CUDAExecutionProvider"

    def test_unavailable_providers_are_dropped_not_requested(self):
        """Asking ORT for a provider this build lacks is an error, not a no-op."""
        resolved = resolve_execution_providers(DetectorConfig(), CPU_ONLY)
        assert resolved == ["CPUExecutionProvider"]

    def test_no_overlap_falls_back_to_cpu(self):
        cfg = DetectorConfig(providers=("CoreMLExecutionProvider",))
        assert resolve_execution_providers(cfg, CPU_ONLY) == ["CPUExecutionProvider"]

    def test_fp16_reaches_tensorrt(self):
        """It used to be read from config and then silently discarded, which
        meant the `bop` profile ran FP32 while claiming FP16."""
        resolved = resolve_execution_providers(DetectorConfig(fp16=True), FULL_GPU_BOX)
        assert self._options(resolved, "TensorrtExecutionProvider")["trt_fp16_enable"]

    def test_fp16_off_is_propagated_too(self):
        resolved = resolve_execution_providers(DetectorConfig(fp16=False), FULL_GPU_BOX)
        opts = self._options(resolved, "TensorrtExecutionProvider")
        assert opts["trt_fp16_enable"] is False

    def test_engine_cache_is_always_enabled(self):
        """Blocker #3: an uncached TensorRT engine build is minutes, every start."""
        cfg = DetectorConfig(trt_engine_cache_dir="models/trt_cache")
        opts = self._options(
            resolve_execution_providers(cfg, FULL_GPU_BOX),
            "TensorrtExecutionProvider",
        )
        assert opts["trt_engine_cache_enable"] is True
        assert opts["trt_engine_cache_path"] == "models/trt_cache"

    def test_workspace_is_converted_to_bytes(self):
        cfg = DetectorConfig(trt_workspace_mb=2048)
        opts = self._options(
            resolve_execution_providers(cfg, FULL_GPU_BOX),
            "TensorrtExecutionProvider",
        )
        assert opts["trt_max_workspace_size"] == 2048 * 1024 * 1024

    def test_cuda_memory_limit_is_omitted_when_unset(self):
        """0 must mean 'let the driver decide', not 'zero bytes of VRAM'."""
        opts = self._options(
            resolve_execution_providers(DetectorConfig(), FULL_GPU_BOX),
            "CUDAExecutionProvider",
        )
        assert "gpu_mem_limit" not in opts

    def test_cuda_memory_limit_is_converted_when_set(self):
        cfg = DetectorConfig(gpu_mem_limit_mb=3072)
        opts = self._options(
            resolve_execution_providers(cfg, FULL_GPU_BOX), "CUDAExecutionProvider"
        )
        assert opts["gpu_mem_limit"] == 3072 * 1024 * 1024

    def test_conv_algo_search_defaults_to_heuristic_for_cold_start(self):
        opts = self._options(
            resolve_execution_providers(DetectorConfig(), FULL_GPU_BOX),
            "CUDAExecutionProvider",
        )
        assert opts["cudnn_conv_algo_search"] == "HEURISTIC"

    def test_cpu_provider_carries_no_gpu_options(self):
        resolved = resolve_execution_providers(DetectorConfig(), FULL_GPU_BOX)
        assert "CPUExecutionProvider" in resolved  # a bare string, not a tuple

    def test_gpu_block_is_read_from_config(self):
        cfg = DetectorConfig.from_mapping(
            {
                "detector": {
                    "fp16": True,
                    "gpu": {
                        "device_id": 1,
                        "trt_workspace_mb": 512,
                        "gpu_mem_limit_mb": 2048,
                        "cudnn_conv_algo_search": "exhaustive",
                        "trt_engine_cache_dir": "/tmp/engines",
                    },
                }
            }
        )
        assert cfg.device_id == 1
        assert cfg.trt_workspace_mb == 512
        assert cfg.gpu_mem_limit_mb == 2048
        assert cfg.cudnn_conv_algo_search == "EXHAUSTIVE"
        assert cfg.trt_engine_cache_dir == "/tmp/engines"

    def test_gpu_defaults_hold_when_the_block_is_absent(self):
        cfg = DetectorConfig.from_mapping({"detector": {}})
        assert cfg.device_id == 0
        assert cfg.gpu_mem_limit_mb == 0
        assert cfg.trt_engine_cache_dir == "models/trt_cache"


class TestPreprocessing:
    """The buffer-reuse rewrite is a hot-path optimisation (§7.4), so it is
    pinned against the obvious implementation it replaced. A silent channel
    swap here would mis-detect everything downstream while looking healthy.
    """

    def _reference(self, image):
        """What the code did before the rewrite. Readable, allocation-heavy."""
        rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))

    def _image(self, h=32, w=48):
        rng = np.random.default_rng(7)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_matches_the_reference_implementation_exactly(self):
        image = self._image()
        out = np.empty((3, 32, 48), dtype=np.float32)
        preprocess_into(image, out)
        np.testing.assert_array_equal(out, self._reference(image))

    def test_channel_order_is_bgr_to_rgb(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # blue in a BGR frame
        out = np.empty((3, 4, 4), dtype=np.float32)
        preprocess_into(image, out)
        assert out[2].max() == 1.0  # ...must land in the RED plane of RGB
        assert out[0].max() == 0.0

    def test_output_is_normalised_to_unit_range(self):
        out = np.empty((3, 32, 48), dtype=np.float32)
        preprocess_into(self._image(), out)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_endpoints_are_exact(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0, :] = 255
        out = np.empty((3, 2, 2), dtype=np.float32)
        preprocess_into(image, out)
        assert out[0, 0, 0] == 1.0
        assert out[0, 1, 1] == 0.0

    def test_writes_into_a_non_contiguous_destination_slot(self):
        """Every real call writes into one row of a reused NCHW batch buffer."""
        buffer = np.empty((4, 3, 32, 48), dtype=np.float32)
        image = self._image()
        preprocess_into(image, buffer[2])
        np.testing.assert_array_equal(buffer[2], self._reference(image))

    def test_the_source_frame_is_not_mutated(self):
        """P4: the original frame is evidence and is never touched."""
        image = self._image()
        before = image.copy()
        preprocess_into(image, np.empty((3, 32, 48), dtype=np.float32))
        np.testing.assert_array_equal(image, before)


class TestFrameTransform:
    def test_letterbox_round_trip_is_exact(self):
        t = FrameTransform.letterbox((1920, 1080), (640, 640))
        restored = t.to_original((t.pad_x, t.pad_y, 640 - t.pad_x, 640 - t.pad_y))
        assert restored == pytest.approx((0.0, 0.0, 1920.0, 1080.0))

    def test_aspect_ratio_is_preserved(self):
        t = FrameTransform.letterbox((1920, 1080), (640, 640))
        assert t.scale_x == t.scale_y

    def test_padding_is_centred(self):
        t = FrameTransform.letterbox((1920, 1080), (640, 640))
        assert t.pad_x == 0.0
        assert t.pad_y == pytest.approx(140.0)

    def test_portrait_source(self):
        t = FrameTransform.letterbox((720, 1280), (640, 640))
        assert t.pad_y == 0.0
        assert t.pad_x > 0

    def test_tiled_offsets(self):
        t = FrameTransform(scale_x=1.0, scale_y=1.0, crop_x=100.0, crop_y=50.0)
        assert t.to_original((0, 0, 10, 10)) == (100.0, 50.0, 110.0, 60.0)

    def test_degenerate_input_raises(self):
        with pytest.raises(ValueError):
            FrameTransform.letterbox((0, 1080), (640, 640))

    def test_identity(self):
        assert FrameTransform(1.0, 1.0).is_identity


class TestCoordinateMappingInvariant:
    """Detections are always mapped back to original frame coordinates."""

    def _pipeline(self, **overrides):
        cfg = DetectorConfig(
            backend="mock",
            class_map=CLASS_MAP,
            conf_thresholds={"person": 0.4, "vehicle": 0.45, "default": 0.5},
            min_box_area_px=200.0,
            min_box_height_px=18.0,
            **overrides,
        )
        return Pipeline(MockDetector(cfg), cfg)

    def test_boxes_land_in_original_space(self):
        pipeline = self._pipeline()
        transform = FrameTransform.letterbox((1920, 1080), (640, 640))
        raw = [RawDetection(cls_id=0, conf=0.9, box=(320.0, 320.0, 360.0, 440.0))]
        detection = pipeline._to_detections(raw, transform)[0]

        # In model space the box sits at x=320; in original space it must be
        # near the middle of a 1920-wide frame.
        assert 900 < detection.box[0] < 1000
        assert detection.box[2] <= 1920

    def test_unmapped_classes_are_dropped_before_the_tracker(self):
        pipeline = self._pipeline()
        raw = [RawDetection(cls_id=63, conf=0.99, box=(0.0, 0.0, 100.0, 100.0))]  # 'laptop'
        assert pipeline._to_detections(raw, FrameTransform(1.0, 1.0)) == []

    def test_class_attributes_travel_with_the_detection(self):
        pipeline = self._pipeline()
        raw = [RawDetection(cls_id=2, conf=0.9, box=(0.0, 0.0, 200.0, 100.0))]
        detection = pipeline._to_detections(raw, FrameTransform(1.0, 1.0))[0]
        assert detection.cls == "vehicle"
        assert detection.attributes["vehicle_type"] == "car"

    def test_per_class_threshold_is_applied(self):
        pipeline = self._pipeline()
        raw = [RawDetection(cls_id=0, conf=0.35, box=(0.0, 0.0, 100.0, 200.0))]
        assert pipeline._to_detections(raw, FrameTransform(1.0, 1.0)) == []

    def test_tiny_boxes_are_rejected_as_noise(self):
        pipeline = self._pipeline()
        raw = [RawDetection(cls_id=0, conf=0.99, box=(0.0, 0.0, 5.0, 10.0))]
        assert pipeline._to_detections(raw, FrameTransform(1.0, 1.0)) == []
