"""End-to-end stage behaviour (§7.14) — the Phase 5 acceptance criteria.

These drive ``CameraWorker._process`` directly with synthetic detection bundles,
which exercises track -> geometry -> rules -> risk -> debounce -> alert without
needing a GPU, a camera or OpenCV.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
from helpers import T0

from ibvap_worker.anpr import AnprConfig, PlateCandidate, plate_hmac
from ibvap_worker.detect import DetectorConfig, MockDetector
from ibvap_worker.enhance import EnhanceConfig
from ibvap_worker.evqm import EVQMConfig
from ibvap_worker.ingest import IngestConfig
from ibvap_worker.pipeline import CameraWorker, DetBundle, Pipeline, _DropOldestQueue
from ibvap_worker.risk import RiskConfig
from ibvap_worker.rules import DebounceConfig, RuleConfig
from ibvap_worker.track import TrackerConfig
from ibvap_worker.types import Detection, Frame, FrameTransform, ZoneKind, ZoneRuntime
from ibvap_worker.watchlist import PlateWatchHit, WatchlistCache


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
        worker = build_worker(camera, [tripwire_zone], recorder, perimeter_approach=False)

        # Walk from x=400 to x=800 across a wire at x=600, over 40 frames.
        for i in range(40):
            worker._process(bundle(camera, i, [person_at(400 + i * 10)]))

        assert len(recorder.alerts) == 1
        alert = recorder.alerts[0]
        assert max(alert["signals"], key=lambda s: s.weight).code == "TRIPWIRE_CROSS"

    def test_crossing_is_not_buried_by_the_earlier_approach_alert(self, camera, tripwire_zone):
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

        kinds = [max(a["signals"], key=lambda s: s.weight).code for a in recorder.alerts]
        assert "PERIMETER_APPROACH" in kinds
        assert "TRIPWIRE_CROSS" in kinds
        # The crossing must be scored higher than the approach that preceded it.
        by_kind = {
            max(a["signals"], key=lambda s: s.weight).code: a["risk"].score for a in recorder.alerts
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

    def test_a_stationary_occluded_person_is_not_treated_as_flicker(self, camera, area_zone):
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
        worker = build_worker(camera, [tripwire_zone], recorder, perimeter_approach=False)
        for i in range(40):
            worker._process(bundle(camera, i, [person_at(400 + i * 10)]))

        risk = recorder.alerts[0]["risk"]
        assert risk.sums_correctly()
        assert 0.0 <= risk.score <= 100.0
        assert risk.severity in ("info", "low", "medium", "high", "critical")

    def test_night_profile_adds_a_reason_code(self, camera, tripwire_zone):
        recorder = Recorder()
        worker = build_worker(camera, [tripwire_zone], recorder, perimeter_approach=False)
        for i in range(40):
            worker._process(bundle(camera, i, [person_at(400 + i * 10)], profile="night"))

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
                        Detection("person", 0.9, (150.0 + i, 200.0, 190.0 + i, 400.0), 0),
                        Detection("person", 0.9, (420.0 - i, 200.0, 460.0 - i, 400.0), 0),
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
        worker = build_worker(camera, [tripwire_zone], recorder, perimeter_approach=False)
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


# ---------------------------------------------------------------------------
# ANPR wiring (§7.9): the pure voting/validation/HMAC logic already has its
# own property tests in test_anpr.py. What matters here is the plumbing --
# that a settled, watchlisted plate actually reaches the rule engine as
# ``plate_hit``, that a non-watchlisted plate does not, and that a broken OCR
# backend degrades the camera, never kills it (P8).
# ---------------------------------------------------------------------------

PLATE_TEXT = "HR26DA1234"
HMAC_KEY = b"x" * 32


class FakeAnprReader:
    """Always reads the same plate, deterministically -- voting is exercised
    for real (min_frames_agreed still has to be met across calls)."""

    def __init__(self, text: str = PLATE_TEXT, raises: bool = False) -> None:
        self.text = text
        self.raises = raises
        self.calls = 0

    def read(self, crop, frame_id):
        self.calls += 1
        if self.raises:
            raise RuntimeError("OCR backend exploded")
        return PlateCandidate(
            raw_text=self.text,
            conf=0.9,
            box=(0.0, 0.0, 10.0, 10.0),
            char_confs=(0.9,) * len(self.text),
            frame_id=frame_id,
        )


def vehicle_at(x: float, conf: float = 0.9) -> Detection:
    return Detection("vehicle", conf, (x - 60.0, 200.0, x + 60.0, 350.0), 2)


def vehicle_zone() -> ZoneRuntime:
    """``area_zone``/``tripwire_zone`` only admit ``person`` -- this is their
    vehicle-shaped twin, since WATCHLIST_PLATE needs a gated vehicle track and
    the rule gate runs after zone/class filtering elsewhere in the engine."""
    return ZoneRuntime(
        zone_id="z-vehicle-area",
        name="Vehicle area",
        kind=ZoneKind.AREA,
        polygon=((100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)),
        classes=("vehicle",),
        severity_base=3,
    )


def bundle_with_image(camera, i: int, detections, profile: str = "day") -> DetBundle:
    """Same as ``bundle()``, but with a real array -- ANPR crops into it."""
    frame = frame_at(camera, i)
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    frame = Frame(frame.camera_id, frame.frame_id, frame.ts_utc, image, camera.width, camera.height)
    return DetBundle(
        frame=frame,
        detections=detections,
        transform=FrameTransform(1.0, 1.0),
        profile=profile,
        enhancement_params={"profile": profile},
        inference_ms=10.0,
    )


def build_anpr_worker(camera, zones, recorder, *, reader, watchlist=None, anpr_cfg=None):
    return CameraWorker(
        camera=camera,
        source="/dev/null/fixture.mp4",
        zones=zones,
        ingest_cfg=IngestConfig(analytics_fps=6.0),
        evqm_cfg=EVQMConfig(enabled=False),
        enhance_cfg=EnhanceConfig(),
        tracker_cfg=TrackerConfig(min_hits=3),
        rule_cfg=RuleConfig(),
        risk_cfg=RiskConfig(),
        debounce_cfg=DebounceConfig(cooldown_s=45.0, escalate_after_s=120.0),
        frame_queue=_DropOldestQueue(4),
        on_alert=recorder,
        anpr_cfg=anpr_cfg
        or AnprConfig(
            enabled=True,
            classes=("vehicle",),
            min_frames_agreed=3,
            # These tests are about the wiring (does a settled plate reach
            # the rule engine, does a bad backend degrade gracefully), not
            # about the frame-cadence throttle -- explicit 1 keeps their
            # frame counts meaning what they say, matching every real read
            # to a frame the way this whole suite was written to expect.
            every_n_frames=1,
        ),
        anpr_reader=reader,
        watchlist=watchlist or WatchlistCache(),
        plate_hmac_key=HMAC_KEY,
    )


class TestAnprWiring:
    def test_settled_watchlisted_plate_reaches_the_rule_engine(self, camera):
        digest = plate_hmac(PLATE_TEXT, HMAC_KEY)
        watchlist = WatchlistCache()
        watchlist.set_plates([PlateWatchHit(digest, "stolen")])
        recorder = Recorder()
        worker = build_anpr_worker(
            camera, [vehicle_zone()], recorder, reader=FakeAnprReader(), watchlist=watchlist
        )

        # >= RuleConfig.min_track_age_frames (8) so the gate lets a standalone
        # rule through at all; the plate itself settles well before that
        # (min_frames_agreed=3), so it is sitting there waiting when it does.
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [vehicle_at(300.0)]))

        assert recorder.alerts, "a watchlisted plate must raise an alert"
        codes = {s.code for a in recorder.alerts for s in a["signals"]}
        assert "WATCHLIST_PLATE" in codes

    def test_settled_plate_not_on_the_watchlist_raises_nothing(self, camera):
        """Reading and voting on a plate is not itself an event -- only a
        watchlist match is (P3: a false alert costs more than a missed one).
        No zones at all, so the only possible alert source is ANPR itself."""
        recorder = Recorder()
        worker = build_anpr_worker(
            camera, [], recorder, reader=FakeAnprReader(), watchlist=WatchlistCache()
        )
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [vehicle_at(300.0)]))
        assert recorder.alerts == []

    def test_a_broken_ocr_backend_degrades_the_camera_not_kills_it(self, camera):
        """P8: one bad crop or a flaky OCR backend must not stop analytics."""
        recorder = Recorder()
        reader = FakeAnprReader(raises=True)
        worker = build_anpr_worker(camera, [vehicle_zone()], recorder, reader=reader)
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [vehicle_at(300.0)]))
        assert reader.calls > 0  # it really was called, and really did raise
        # No crash reached this line; the vehicle's OWN zone-intrusion alert
        # (unrelated to ANPR) still fires normally.
        assert any(any(s.code == "ZONE_INTRUSION" for s in a["signals"]) for a in recorder.alerts)

    def test_disabled_anpr_never_calls_the_reader(self, camera):
        reader = FakeAnprReader()
        worker = build_anpr_worker(
            camera,
            [vehicle_zone()],
            Recorder(),
            reader=reader,
            anpr_cfg=AnprConfig(enabled=False),
        )
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [vehicle_at(300.0)]))
        assert reader.calls == 0

    def test_a_closed_track_does_not_leak_its_voter(self, camera):
        """The plate voter dict is per-track state living outside the tracker;
        it must be torn down at close_expired like everything else (§7.5)."""
        reader = FakeAnprReader()
        worker = build_anpr_worker(camera, [vehicle_zone()], Recorder(), reader=reader)
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [vehicle_at(300.0)]))
        assert worker._plate_voters  # settled during the run
        # Push enough frames with no detections that the tracker expires it.
        for i in range(10, 45):
            worker._process(bundle_with_image(camera, i, []))
        assert worker._plate_voters == {}


class TestFrameSizeReconciliation:
    """Zones are stored normalised and denormalised once at startup against the
    camera row's ``resolution_w/h``. Nothing updates that column when an
    operator binds a real camera by IP, so the decoded frame — not the row —
    has to be the authority on geometry, or every zone lands in the wrong
    place on any camera that is not the seeded 1280x720.
    """

    def test_zones_rescale_to_the_size_the_camera_actually_streams(self, camera, area_zone):
        worker = build_worker(camera, [area_zone], Recorder())
        worker._on_frame(
            Frame(
                camera_id=camera.camera_id,
                frame_id=1,
                ts_utc=T0,
                image=None,
                width=640,
                height=360,
            )
        )
        assert (worker.camera.width, worker.camera.height) == (640, 360)
        # Half the configured size in both axes, so every vertex halves.
        assert worker.zones[0].polygon == (
            (50.0, 50.0),
            (250.0, 50.0),
            (250.0, 250.0),
            (50.0, 250.0),
        )

    def test_a_portrait_stream_rescales_each_axis_independently(self, camera, area_zone):
        """The failure this guards is a phone held upright: the aspect ratio
        inverts, so a single uniform scale factor would still be wrong."""
        worker = build_worker(camera, [area_zone], Recorder())
        worker._on_frame(
            Frame(
                camera_id=camera.camera_id,
                frame_id=1,
                ts_utc=T0,
                image=None,
                width=720,
                height=1280,
            )
        )
        sx, sy = 720 / 1280, 1280 / 720
        assert worker.zones[0].polygon[1] == (500.0 * sx, 100.0 * sy)

    def test_a_matching_frame_leaves_geometry_untouched(self, camera, area_zone):
        worker = build_worker(camera, [area_zone], Recorder())
        worker._on_frame(frame_at(camera, 1))
        assert worker.zones[0] is area_zone


def build_gesture_worker(camera, zones, recorder, *, estimator, gesture_cfg=None):
    from ibvap_worker.gesture import GestureConfig

    return CameraWorker(
        camera=camera,
        source="/dev/null/fixture.mp4",
        zones=zones,
        ingest_cfg=IngestConfig(analytics_fps=6.0),
        evqm_cfg=EVQMConfig(enabled=False),
        enhance_cfg=EnhanceConfig(),
        tracker_cfg=TrackerConfig(min_hits=3),
        rule_cfg=RuleConfig(),
        risk_cfg=RiskConfig(),
        debounce_cfg=DebounceConfig(cooldown_s=45.0, escalate_after_s=120.0),
        frame_queue=_DropOldestQueue(4),
        on_alert=recorder,
        gesture_cfg=gesture_cfg
        or GestureConfig(enabled=True, every_n_frames=1, min_frames_agreed=4, window_frames=8),
        pose_estimator=estimator,
    )


class TestHandSignals:
    """§7.7 gestures, driven through the real pipeline with a mock pose model.
    'Mock the GPU, not the logic.'"""

    def _surrender_pose(self):
        """A person with both hands above their shoulders.

        These are MODEL-space keypoints, which is what a real backend returns;
        the pipeline maps them back through the crop's FrameTransform before
        classifying. That mapping is a uniform scale plus a translation, and
        the classifier is invariant to both (test_gesture.py proves it), so the
        gesture survives the round trip -- which is the point of having built
        it that way.
        """
        from ibvap_worker.detect.onnx_pose import MockPoseEstimator
        from ibvap_worker.gesture import COCO_KEYPOINTS, Keypoint, Pose

        joints = {
            "nose": (330.0, 240.0),
            "left_eye": (325.0, 235.0),
            "right_eye": (335.0, 235.0),
            "left_ear": (320.0, 238.0),
            "right_ear": (340.0, 238.0),
            "left_shoulder": (310.0, 260.0),
            "right_shoulder": (350.0, 260.0),
            "left_elbow": (308.0, 235.0),
            "right_elbow": (352.0, 235.0),
            "left_wrist": (308.0, 210.0),
            "right_wrist": (352.0, 210.0),
            "left_hip": (315.0, 360.0),
            "right_hip": (345.0, 360.0),
            "left_knee": (315.0, 430.0),
            "right_knee": (345.0, 430.0),
            "left_ankle": (315.0, 500.0),
            "right_ankle": (345.0, 500.0),
        }
        pose = Pose(tuple(Keypoint(*joints[n], 0.9) for n in COCO_KEYPOINTS))
        return MockPoseEstimator([pose])

    def test_a_held_signal_enriches_a_real_alert(self, camera, area_zone):
        """The whole chain: crop -> pose -> map back to original coordinates ->
        classify -> vote -> signal -> risk breakdown.

        The person stands inside an area zone with their hands up. ZONE_INTRUSION
        is what raises the alert (HAND_SIGNAL is contextual and cannot); HANDS_UP
        has to be riding along in the breakdown, carrying its negative weight."""
        estimator = self._surrender_pose()
        recorder = Recorder()
        worker = build_gesture_worker(camera, [area_zone], recorder, estimator=estimator)
        for i in range(12):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))

        assert estimator.calls > 0
        assert recorder.alerts, "the zone intrusion itself should have alerted"
        codes = {s.code for a in recorder.alerts for s in a["signals"]}
        assert "ZONE_INTRUSION" in codes
        assert "HANDS_UP" in codes, f"gesture never reached the alert; got {codes}"

        # P3: showing empty hands must REDUCE the score, and the breakdown must
        # still sum to it (P2).
        alert = next(a for a in recorder.alerts if any(s.code == "HANDS_UP" for s in a["signals"]))
        hands_up = next(s for s in alert["signals"] if s.code == "HANDS_UP")
        assert hands_up.weight < 0
        assert alert["risk"].sums_correctly()

    def test_contextual_only_never_raises_on_its_own(self, camera):
        """A person waving in an empty field is not an alert. With no zones at
        all, the gesture must produce nothing -- blocker #1."""
        estimator = self._surrender_pose()
        recorder = Recorder()
        worker = build_gesture_worker(camera, [], recorder, estimator=estimator)
        for i in range(14):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert estimator.calls > 0
        assert recorder.alerts == []

    def test_disabled_costs_nothing(self, camera, area_zone):
        from ibvap_worker.gesture import GestureConfig

        estimator = self._surrender_pose()
        worker = build_gesture_worker(
            camera,
            [area_zone],
            Recorder(),
            estimator=estimator,
            gesture_cfg=GestureConfig(enabled=False),
        )
        for i in range(8):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert estimator.calls == 0

    def test_a_closed_track_does_not_leak_its_voter(self, camera, area_zone):
        estimator = self._surrender_pose()
        worker = build_gesture_worker(camera, [area_zone], Recorder(), estimator=estimator)
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert worker._gesture_voters
        for i in range(10, 45):
            worker._process(bundle_with_image(camera, i, []))
        assert worker._gesture_voters == {}

    def test_a_broken_pose_backend_never_costs_the_frame(self, camera, area_zone):
        """P8: a flaky accelerator must not take the camera down with it."""

        class Exploding:
            input_size = (640, 640)

            def estimate(self, _canvas):
                raise RuntimeError("CUDA fell over")

        recorder = Recorder()
        worker = build_gesture_worker(camera, [area_zone], recorder, estimator=Exploding())
        for i in range(12):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        # The zone intrusion still alerted, despite pose failing on every frame.
        assert recorder.alerts


def build_weapon_worker(camera, zones, recorder, *, detector, weapon_cfg=None):
    from ibvap_worker.weapon import WeaponConfig

    return CameraWorker(
        camera=camera,
        source="/dev/null/fixture.mp4",
        zones=zones,
        ingest_cfg=IngestConfig(analytics_fps=6.0),
        evqm_cfg=EVQMConfig(enabled=False),
        enhance_cfg=EnhanceConfig(),
        tracker_cfg=TrackerConfig(min_hits=3),
        rule_cfg=RuleConfig(),
        risk_cfg=RiskConfig(),
        debounce_cfg=DebounceConfig(cooldown_s=45.0, escalate_after_s=120.0),
        frame_queue=_DropOldestQueue(4),
        on_alert=recorder,
        weapon_cfg=weapon_cfg
        or WeaponConfig(enabled=True, every_n_frames=1, min_frames_agreed=3, window_frames=8),
        weapon_detector=detector,
    )


class TestWeaponDetection:
    """§7.7 weapons, driven through the real pipeline with a mock detector."""

    def _armed(self, conf=0.9):
        from ibvap_worker.detect.onnx_weapon import MockWeaponDetector
        from ibvap_worker.weapon import WeaponCandidate

        return MockWeaponDetector([WeaponCandidate("guns", conf)])

    def test_an_armed_person_raises_an_alert_with_no_zone_at_all(self, camera):
        """WEAPON_VISIBLE is STANDALONE, unlike hand signals. A person carrying
        a firearm is an event before they cross anything -- waiting for a zone
        breach would be waiting for the thing the alert exists to prevent."""
        detector = self._armed()
        recorder = Recorder()
        worker = build_weapon_worker(camera, [], recorder, detector=detector)
        for i in range(14):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))

        assert detector.calls > 0
        assert recorder.alerts, "an armed person with no zone should still alert"
        codes = {s.code for a in recorder.alerts for s in a["signals"]}
        assert "WEAPON_VISIBLE" in codes

    def test_it_scores_high_even_on_a_young_track(self, camera):
        """Regression: WEAPON_VISIBLE has to outrank the SHORT_TRACK discount.

        The alert fires the moment the rule gate opens, which is also while the
        track is still young enough for SHORT_TRACK (-12) to apply. At a weight
        of 60 that landed an armed person in the queue as 'medium' -- and
        severity is what an operator triages on. The discount is still applied
        and still visible in the breakdown; it just no longer decides the band.
        """
        recorder = Recorder()
        worker = build_weapon_worker(camera, [], recorder, detector=self._armed())
        for i in range(14):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))

        alert = recorder.alerts[0]
        codes = {s.code for s in alert["risk"].breakdown}
        assert "SHORT_TRACK" in codes, "the honesty discount must still be applied"
        assert alert["risk"].severity in ("high", "critical")
        assert alert["risk"].sums_correctly()  # P2

    def test_an_unarmed_person_never_fires(self, camera):
        from ibvap_worker.detect.onnx_weapon import MockWeaponDetector

        detector = MockWeaponDetector([None])
        recorder = Recorder()
        worker = build_weapon_worker(camera, [], recorder, detector=detector)
        for i in range(14):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert detector.calls > 0
        assert recorder.alerts == []

    def test_disabled_costs_nothing(self, camera):
        from ibvap_worker.weapon import WeaponConfig

        detector = self._armed()
        worker = build_weapon_worker(
            camera, [], Recorder(), detector=detector, weapon_cfg=WeaponConfig(enabled=False)
        )
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert detector.calls == 0

    def test_a_closed_track_does_not_leak_its_voter(self, camera):
        worker = build_weapon_worker(camera, [], Recorder(), detector=self._armed())
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert worker._weapon_voters
        for i in range(10, 45):
            worker._process(bundle_with_image(camera, i, []))
        assert worker._weapon_voters == {}

    def test_a_broken_backend_never_costs_the_camera(self, camera, area_zone):
        """P8: a flaky accelerator must not take the pipeline down with it."""

        class Exploding:
            input_size = (640, 640)

            def detect(self, _canvas):
                raise RuntimeError("CUDA fell over")

        recorder = Recorder()
        worker = build_weapon_worker(camera, [area_zone], recorder, detector=Exploding())
        for i in range(12):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        # The zone intrusion still alerted despite weapon detection failing.
        assert recorder.alerts


class TestActivityGating:
    """§7.7 activity gate: skip the expensive models when nothing changed.

    The saving is the easy half. The half that matters is that stillness must
    never clear a confirmed state -- an armed person who stops moving is still
    armed, and a naive gate would quietly disarm them.
    """

    def _still_bundle(self, camera, i, dets):
        """A bundle whose image is IDENTICAL every frame, so the gate sees no
        motion at all. bundle_with_image() already makes a constant zero image,
        which is exactly the pathological 'nothing is changing' input."""
        return bundle_with_image(camera, i, dets)

    def test_a_still_crop_stops_costing_inferences(self, camera):
        from ibvap_worker.detect.onnx_weapon import MockWeaponDetector
        from ibvap_worker.weapon import WeaponCandidate, WeaponConfig

        detector = MockWeaponDetector([WeaponCandidate("guns", 0.9)])
        worker = build_weapon_worker(
            camera,
            [],
            Recorder(),
            detector=detector,
            weapon_cfg=WeaponConfig(
                enabled=True,
                every_n_frames=1,
                min_frames_agreed=3,
                window_frames=8,
                activity_gate={"max_stale_frames": 1000},
            ),
        )
        for i in range(30):
            worker._process(self._still_bundle(camera, i, [person_at(300.0)]))

        stats = worker._weapon_gate.stats()
        assert stats["skips"] > 0, "a motionless crop should skip inferences"
        # Far fewer model calls than frames processed.
        assert detector.calls < 30

    def test_an_armed_person_who_stops_moving_stays_armed(self, camera):
        """THE safety property. The gate decides whether to spend an inference,
        never that a previous answer expired."""
        from ibvap_worker.detect.onnx_weapon import MockWeaponDetector
        from ibvap_worker.weapon import WeaponCandidate, WeaponConfig

        recorder = Recorder()
        detector = MockWeaponDetector([WeaponCandidate("guns", 0.9)])
        worker = build_weapon_worker(
            camera,
            [],
            recorder,
            detector=detector,
            weapon_cfg=WeaponConfig(
                enabled=True,
                every_n_frames=1,
                min_frames_agreed=3,
                window_frames=8,
                activity_gate={"max_stale_frames": 1000},
            ),
        )
        # Long enough that the gate is skipping most frames by the end.
        for i in range(40):
            worker._process(self._still_bundle(camera, i, [person_at(300.0)]))

        assert worker._weapon_gate.stats()["skips"] > 0
        assert recorder.alerts, "the armed person must still have alerted"
        codes = {s.code for a in recorder.alerts for s in a["signals"]}
        assert "WEAPON_VISIBLE" in codes
        # And the held state is what kept reporting it while the crop was still.
        assert worker._weapon_last, "a confirmed weapon must be retained across skips"

    def test_putting_the_weapon_down_still_clears_it(self, camera):
        """The other direction: a real re-check that comes back negative must
        clear the held state, or nobody could ever stop being armed."""
        from ibvap_worker.detect.onnx_weapon import MockWeaponDetector
        from ibvap_worker.weapon import WeaponCandidate, WeaponConfig

        # Armed for the first few calls, then clean for the rest.
        detector = MockWeaponDetector([WeaponCandidate("guns", 0.9)] * 4 + [None] * 40)
        worker = build_weapon_worker(
            camera,
            [],
            Recorder(),
            detector=detector,
            weapon_cfg=WeaponConfig(
                enabled=True,
                every_n_frames=1,
                min_frames_agreed=3,
                window_frames=4,
                # Heartbeat every frame, so the model really is re-consulted
                # even though the synthetic image never changes.
                activity_gate={"max_stale_frames": 1},
            ),
        )
        for i in range(30):
            worker._process(self._still_bundle(camera, i, [person_at(300.0)]))
        assert worker._weapon_last == {}, "a negative re-check must clear the held state"

    def test_the_gate_state_does_not_leak_on_track_close(self, camera):
        from ibvap_worker.detect.onnx_weapon import MockWeaponDetector
        from ibvap_worker.weapon import WeaponCandidate

        worker = build_weapon_worker(
            camera, [], Recorder(), detector=MockWeaponDetector([WeaponCandidate("guns", 0.9)])
        )
        for i in range(10):
            worker._process(self._still_bundle(camera, i, [person_at(300.0)]))
        for i in range(10, 45):
            worker._process(self._still_bundle(camera, i, []))
        assert worker._weapon_last == {}
        assert worker._weapon_gate._thumbs == {}
