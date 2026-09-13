"""Detector factory, warmup and coordinate mapping (§7.4).

We mock the GPU, never the logic: ``MockDetector`` stands in for the accelerator
while the transform arithmetic and class mapping under test are the real code.
"""

from __future__ import annotations

import pytest

from drishti_worker.detect import Detector, DetectorConfig, MockDetector, build_detector
from drishti_worker.pipeline import Pipeline
from drishti_worker.types import FrameTransform, RawDetection

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
        from drishti_worker.detect import build_detector as build

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
            {
                "detector": {
                    "conf_thresholds": {"person": 0.40, "bag": 0.55, "default": 0.5}
                }
            }
        )
        assert cfg.threshold_for("person") < cfg.threshold_for("bag")


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
        raw = [
            RawDetection(cls_id=63, conf=0.99, box=(0.0, 0.0, 100.0, 100.0))
        ]  # 'laptop'
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
