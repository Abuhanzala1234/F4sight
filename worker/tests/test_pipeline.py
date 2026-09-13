"""End-to-end stage behaviour (§7.14) — the Phase 5 acceptance criteria.

These drive ``CameraWorker._process`` directly with synthetic detection bundles,
which exercises track -> geometry -> rules -> risk -> debounce -> alert without
needing a GPU, a camera or OpenCV.
"""

from __future__ import annotations

from datetime import timedelta

from helpers import T0

from drishti_worker.detect import DetectorConfig, MockDetector
from drishti_worker.enhance import EnhanceConfig
from drishti_worker.evqm import EVQMConfig
from drishti_worker.ingest import IngestConfig
from drishti_worker.pipeline import CameraWorker, DetBundle, Pipeline, _DropOldestQueue
from drishti_worker.risk import RiskConfig
from drishti_worker.rules import DebounceConfig, RuleConfig
from drishti_worker.track import TrackerConfig
from drishti_worker.types import Detection, Frame, FrameTransform


class Recorder:
    def __init__(self):
        self.alerts = []

    def __call__(self, **kwargs):
        self.alerts.append(kwargs)


def build_worker(camera, zones, recorder, **rule_overrides):
    cfg = RuleConfig(**rule_overrides) if rule_overrides else RuleConfig()
    return CameraWorker(
        camera=camera,
        source="/dev/null/fixture.mp4",
        zones=zones,
        ingest_cfg=IngestConfig(analytics_fps=6.0),
        evqm_cfg=EVQMConfig(enabled=False),
        enhance_cfg=EnhanceConfig(),
        tracker_cfg=TrackerConfig(min_hits=3),
        rule_cfg=cfg,
        risk_cfg=RiskConfig(),
        debounce_cfg=DebounceConfig(cooldown_s=45.0, escalate_after_s=120.0),
        frame_queue=_DropOldestQueue(4),
        on_alert=recorder,
    )


def frame_at(camera, i: int) -> Frame:
    return Frame(
        camera_id=camera.camera_id,
        frame_id=i,
        ts_utc=T0 + timedelta(seconds=i / 6.0),
        image=None,
        width=camera.width,
        height=camera.height,
    )


def bundle(camera, i: int, detections, profile: str = "day") -> DetBundle:
    return DetBundle(
        frame=frame_at(camera, i),
        detections=detections,
        transform=FrameTransform(1.0, 1.0),
        profile=profile,
        enhancement_params={"profile": profile},
        inference_ms=10.0,
    )


def person_at(x: float, conf: float = 0.9) -> Detection:
    return Detection("person", conf, (x - 20.0, 200.0, x + 20.0, 400.0), 0)


class TestIntrusionEndToEnd:
    def test_one_crossing_produces_exactly_one_alert(self, camera, tripwire_zone):
        """Phase 5 acceptance: a person walking across a tripwire yields exactly
        one alert per crossing, not one per frame.

        Early-warning is disabled here so the crossing is isolated; the
        approach-then-cross interaction has its own test below.
        """
        recorder = Recorder()
        worker = build_worker(
            camera, [tripwire_zone], recorder, perimeter_approach=False
        )

        # Walk from x=400 to x=800 across a wire at x=600, over 40 frames.
        for i in range(40):
            worker._process(bundle(camera, i, [person_at(400 + i * 10)]))

        assert len(recorder.alerts) == 1
        alert = recorder.alerts[0]
        assert max(alert["signals"], key=lambda s: s.weight).code == "TRIPWIRE_CROSS"

    def test_crossing_is_not_buried_by_the_earlier_approach_alert(
        self, camera, tripwire_zone
    ):
        """Correlation must never hide an escalation.

        The same person triggers PERIMETER_APPROACH (+18) and then, a second
        later, TRIPWIRE_CROSS inbound (+45). Merging the second into the first
        would show an operator "PERIMETER_APPROACH, low" for an actual
        intrusion - and severity is what they triage on. P2/P3.
        """
        recorder = Recorder()
        worker = build_worker(camera, [tripwire_zone], recorder)
        for i in range(40):
            worker._process(bundle(camera, i, [person_at(400 + i * 10)]))

        kinds = [
            max(a["signals"], key=lambda s: s.weight).code for a in recorder.alerts
        ]
        assert "PERIMETER_APPROACH" in kinds
        assert "TRIPWIRE_CROSS" in kinds
        # The crossing must be scored higher than the approach that preceded it.
        by_kind = {
            max(a["signals"], key=lambda s: s.weight).code: a["risk"].score
            for a in recorder.alerts
        }
        assert by_kind["TRIPWIRE_CROSS"] > by_kind["PERIMETER_APPROACH"]

    def test_idle_scene_produces_no_alerts(self, camera, tripwire_zone, area_zone):
        """Phase 5 acceptance: a long idle fixture yields zero alerts. This is
        the number an operator actually judges the system by (P3)."""
        recorder = Recorder()
        worker = build_worker(camera, [tripwire_zone, area_zone], recorder)
        for i in range(3600):  # ten minutes at 6 fps
            worker._process(bundle(camera, i, []))
        assert recorder.alerts == []

    def test_detector_flicker_never_reaches_the_rules(self, camera, area_zone):
        """Real detector flicker is isolated ghosts at scattered positions.
        They never associate into a track, so min_track_age gates them out
        before any rule sees them (§7.7.1).

        Note the contrast with the next test: a ghost that reappears at the
        SAME place is indistinguishable from a person standing behind a bush,
        and the tracker is right to keep that alive."""
        import random

        rng = random.Random(7)
        recorder = Recorder()
        worker = build_worker(camera, [area_zone], recorder)
        for i in range(120):
            detections = []
            if i % 7 == 0:
                x = rng.uniform(120.0, 480.0)
                y = rng.uniform(150.0, 480.0)
                detections = [Detection("person", 0.6, (x, y, x + 30.0, y + 70.0), 0)]
            worker._process(bundle(camera, i, detections))
        assert recorder.alerts == []

    def test_a_stationary_occluded_person_is_not_treated_as_flicker(
        self, camera, area_zone
    ):
        """The other side of the same coin: intermittent detections at one spot
        keep their track id and DO raise, which is what saves an intrusion
        behind a fence post."""
        recorder = Recorder()
        worker = build_worker(camera, [area_zone], recorder)
        for i in range(60):
            detections = [person_at(300)] if i % 3 == 0 else []
            worker._process(bundle(camera, i, detections))
        assert len(recorder.alerts) >= 1

    def test_risk_breakdown_sums_to_score(self, camera, tripwire_zone):
        """P2, checked on a real alert that came through the whole pipeline."""
        recorder = Recorder()
        worker = build_worker(
            camera, [tripwire_zone], recorder, perimeter_approach=False
        )
        for i in range(40):
            worker._process(bundle(camera, i, [person_at(400 + i * 10)]))

        risk = recorder.alerts[0]["risk"]
        assert risk.sums_correctly()
        assert 0.0 <= risk.score <= 100.0
        assert risk.severity in ("info", "low", "medium", "high", "critical")

    def test_night_profile_adds_a_reason_code(self, camera, tripwire_zone):
        recorder = Recorder()
        worker = build_worker(
            camera, [tripwire_zone], recorder, perimeter_approach=False
        )
        for i in range(40):
            worker._process(
                bundle(camera, i, [person_at(400 + i * 10)], profile="night")
            )

        codes = recorder.alerts[0]["risk"].reason_codes()
        assert "NIGHT_MOVEMENT" in codes
        assert "TRIPWIRE_CROSS" in codes

    def test_two_intruders_produce_two_alerts(self, camera, area_zone):
        recorder = Recorder()
        worker = build_worker(camera, [area_zone], recorder)
        for i in range(30):
            # Two separate people, well apart, both entering the area.
            worker._process(
                bundle(
                    camera,
                    i,
                    [
                        Detection(
                            "person", 0.9, (150.0 + i, 200.0, 190.0 + i, 400.0), 0
                        ),
                        Detection(
                            "person", 0.9, (420.0 - i, 200.0, 460.0 - i, 400.0), 0
                        ),
                    ],
                )
            )
        assert len(recorder.alerts) == 2
        assert len({a["track"].track_id for a in recorder.alerts}) == 2

    def test_masked_region_is_silent(self, camera, area_zone, mask_zone):
        recorder = Recorder()
        worker = build_worker(camera, [area_zone, mask_zone], recorder)
        for i in range(40):
            # Inside the area polygon but also inside the mask.
            worker._process(
                bundle(
                    camera,
                    i,
                    [Detection("person", 0.9, (280.0, 620.0, 320.0, 700.0), 0)],
                )
            )
        assert recorder.alerts == []

    def test_stats_are_updated(self, camera, tripwire_zone):
        recorder = Recorder()
        worker = build_worker(
            camera, [tripwire_zone], recorder, perimeter_approach=False
        )
        for i in range(40):
            worker._process(bundle(camera, i, [person_at(400 + i * 10)]))
        assert worker.stats.alerts_emitted == 1

    def test_health_is_reportable(self, camera, tripwire_zone):
        worker = build_worker(camera, [tripwire_zone], Recorder())
        health = worker.health()
        assert health["camera_code"] == "CAM-01"
        assert "evqm_profile" in health
        assert "ingest" in health


class TestBackpressure:
    def test_frame_queue_drops_the_oldest(self):
        """On a slow host we analyse RECENT frames. A backlog of stale frames
        produces alerts about where someone was thirty seconds ago."""
        q = _DropOldestQueue(3)
        for i in range(10):
            q.put_latest(i)
        assert [q.get() for _ in range(3)] == [7, 8, 9]
        assert q.dropped == 7

    def test_drops_are_counted_for_honest_reporting(self):
        q = _DropOldestQueue(2)
        for i in range(5):
            q.put_latest(i)
        assert q.dropped == 3

    def test_queue_below_capacity_never_drops(self):
        q = _DropOldestQueue(10)
        for i in range(5):
            q.put_latest(i)
        assert q.dropped == 0


class TestPipelineAssembly:
    def test_health_payload(self):
        cfg = DetectorConfig(backend="mock")
        pipeline = Pipeline(MockDetector(cfg), cfg)
        health = pipeline.health()
        assert health["detector"] == "mock"
        assert health["cameras"] == []
        assert "frames_dropped_queue" in health

    def test_workers_are_registered(self, camera, tripwire_zone):
        cfg = DetectorConfig(backend="mock")
        pipeline = Pipeline(MockDetector(cfg), cfg)
        pipeline.add_worker(build_worker(camera, [tripwire_zone], Recorder()))
        assert len(pipeline.health()["cameras"]) == 1
