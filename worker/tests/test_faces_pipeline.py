"""Face matching driven through the real pipeline (§7.10). OPT-IN.

Kept in its own file rather than appended to test_pipeline.py because face
matching is the one stage whose tests are mostly about refusal: naming a
specific human being is the highest-stakes false positive in this system
(P3, P6), so most of what matters here is what must NOT produce a name.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
from helpers import T0

from ibvap_worker.detect.onnx_face import MockFaceDetector, MockFaceEmbedder
from ibvap_worker.enhance import EnhanceConfig
from ibvap_worker.evqm import EVQMConfig
from ibvap_worker.faces import FaceConfig, FaceDetection, WatchlistFace
from ibvap_worker.ingest import IngestConfig
from ibvap_worker.pipeline import CameraWorker, DetBundle, _DropOldestQueue
from ibvap_worker.risk import RiskConfig
from ibvap_worker.rules import DebounceConfig, RuleConfig
from ibvap_worker.track import TrackerConfig
from ibvap_worker.types import Detection, Frame, FrameTransform
from ibvap_worker.watchlist import FaceWatchlistCache


class Recorder:
    def __init__(self):
        self.alerts = []

    def __call__(self, **kwargs):
        self.alerts.append(kwargs)


def person_at(x: float, conf: float = 0.9) -> Detection:
    return Detection("person", conf, (x - 20.0, 200.0, x + 20.0, 400.0), 0)


def bundle_with_image(camera, i: int, detections, profile: str = "day") -> DetBundle:
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    frame = Frame(
        camera_id=camera.camera_id,
        frame_id=i,
        ts_utc=T0 + timedelta(seconds=i / 6.0),
        image=image,
        width=camera.width,
        height=camera.height,
    )
    return DetBundle(
        frame=frame,
        detections=detections,
        transform=FrameTransform(1.0, 1.0),
        profile=profile,
        enhancement_params={"profile": profile},
        inference_ms=10.0,
    )


def build_face_worker(camera, zones, recorder, *, detector, embedder, watchlist, face_cfg=None):
    return CameraWorker(
        camera=camera,
        source="/dev/null/fixture.mp4",
        zones=zones,
        ingest_cfg=IngestConfig(analytics_fps=6.0),
        evqm_cfg=EVQMConfig(enabled=False),
        enhance_cfg=EnhanceConfig(),
        tracker_cfg=TrackerConfig(min_hits=3),
        rule_cfg=RuleConfig(watchlist_face=True),
        risk_cfg=RiskConfig(),
        debounce_cfg=DebounceConfig(cooldown_s=45.0, escalate_after_s=120.0),
        frame_queue=_DropOldestQueue(4),
        on_alert=recorder,
        face_cfg=face_cfg
        or FaceConfig(enabled=True, every_n_frames=1, min_frames_agreed=2, window_frames=6),
        face_detector=detector,
        face_embedder=embedder,
        face_watchlist=watchlist,
    )


def make_parts(*, embedding=None, face_score=0.9, face_px=80):
    """A detector that always finds one usable face, an embedder returning a
    fixed vector, and a watchlist holding that same person."""
    emb = embedding or tuple([1.0] + [0.0] * 511)
    face = FaceDetection(
        box=(10.0, 10.0, 10.0 + face_px, 10.0 + face_px),
        score=face_score,
        landmarks=((30.0, 40.0), (60.0, 40.0), (45.0, 55.0), (33.0, 70.0), (57.0, 70.0)),
    )
    detector = MockFaceDetector([[face]])
    embedder = MockFaceEmbedder([emb])
    watchlist = FaceWatchlistCache(threshold=0.55)
    watchlist.set_faces([WatchlistFace("p1", "REF-1", "watch", emb)])
    return detector, embedder, watchlist


class TestFaceMatching:
    def test_a_watchlist_face_raises_an_alert(self, camera):
        detector, embedder, watchlist = make_parts()
        recorder = Recorder()
        worker = build_face_worker(
            camera, [], recorder, detector=detector, embedder=embedder, watchlist=watchlist
        )
        for i in range(14):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))

        assert detector.calls > 0
        assert recorder.alerts, "an enrolled face should raise on its own (standalone rule)"
        codes = {s.code for a in recorder.alerts for s in a["signals"]}
        assert "WATCHLIST_FACE" in codes
        hit = next(s for a in recorder.alerts for s in a["signals"] if s.code == "WATCHLIST_FACE")
        assert hit.detail["person_ref"] == "REF-1"

    def test_a_stranger_never_gets_named(self, camera):
        """An embedding matching nobody must produce no alert at all -- not a
        weak one, not a 'possible' one."""
        detector, embedder, watchlist = make_parts()
        embedder.embeddings = [tuple([0.0, 1.0] + [0.0] * 510)]
        recorder = Recorder()
        worker = build_face_worker(
            camera, [], recorder, detector=detector, embedder=embedder, watchlist=watchlist
        )
        for i in range(14):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert detector.calls > 0
        assert recorder.alerts == []

    def test_an_empty_watchlist_short_circuits_before_any_inference(self, camera):
        """P6: with nobody enrolled there is nothing to match against, so the
        models must not run at all -- no embedding should even be computed."""
        detector, embedder, _ = make_parts()
        empty = FaceWatchlistCache(threshold=0.55)
        worker = build_face_worker(
            camera, [], Recorder(), detector=detector, embedder=embedder, watchlist=empty
        )
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert detector.calls == 0
        assert embedder.calls == 0

    def test_disabled_costs_nothing(self, camera):
        detector, embedder, watchlist = make_parts()
        worker = build_face_worker(
            camera,
            [],
            Recorder(),
            detector=detector,
            embedder=embedder,
            watchlist=watchlist,
            face_cfg=FaceConfig(enabled=False),
        )
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert detector.calls == 0
        assert embedder.calls == 0

    def test_a_face_too_small_to_trust_is_not_matched(self, camera):
        """Matching a 12-px face is not recognition, it is a coin toss with a
        person's name attached (min_face_px)."""
        detector, embedder, watchlist = make_parts(face_px=12)
        recorder = Recorder()
        worker = build_face_worker(
            camera, [], recorder, detector=detector, embedder=embedder, watchlist=watchlist
        )
        for i in range(14):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert detector.calls > 0
        assert embedder.calls == 0, "a too-small face must never reach the embedder"
        assert recorder.alerts == []

    def test_a_matched_person_who_stops_moving_stays_matched(self, camera):
        """Same safety property the weapon path has: the activity gate decides
        whether to spend an inference, never that a previous answer expired."""
        detector, embedder, watchlist = make_parts()
        worker = build_face_worker(
            camera, [], Recorder(), detector=detector, embedder=embedder, watchlist=watchlist
        )
        for i in range(40):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))

        assert worker._face_gate.stats()["skips"] > 0, "a still crop should skip inferences"
        assert worker._face_last, "a confirmed match must be retained across skips"

    def test_a_broken_backend_never_costs_the_camera(self, camera, area_zone):
        """P8: a flaky accelerator must not take the pipeline down with it."""

        class Exploding:
            input_size = (640, 640)

            def detect(self, _canvas):
                raise RuntimeError("CUDA fell over")

        _d, embedder, watchlist = make_parts()
        recorder = Recorder()
        worker = build_face_worker(
            camera,
            [area_zone],
            recorder,
            detector=Exploding(),
            embedder=embedder,
            watchlist=watchlist,
        )
        for i in range(12):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        # The zone intrusion still alerted despite face matching failing.
        assert recorder.alerts

    def test_a_closed_track_does_not_leak_face_state(self, camera):
        """P6: per-track face state, including any embedding-derived result,
        is destroyed with the track (§7.5)."""
        detector, embedder, watchlist = make_parts()
        worker = build_face_worker(
            camera, [], Recorder(), detector=detector, embedder=embedder, watchlist=watchlist
        )
        for i in range(10):
            worker._process(bundle_with_image(camera, i, [person_at(300.0)]))
        assert worker._face_voters
        for i in range(10, 45):
            worker._process(bundle_with_image(camera, i, []))
        assert worker._face_voters == {}
        assert worker._face_last == {}
        assert worker._face_gate._thumbs == {}
