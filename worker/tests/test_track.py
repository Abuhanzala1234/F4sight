"""ByteTrack (§7.5). The second association pass is the thing under test."""

from __future__ import annotations

from datetime import timedelta

import pytest
from helpers import T0

from drishti_worker.track import ByteTracker, TrackerConfig
from drishti_worker.types import Detection


def person(x: float, conf: float = 0.9) -> Detection:
    return Detection("person", conf, (x, 200.0, x + 40.0, 320.0), 0)


def at(i: int):
    return T0 + timedelta(seconds=i / 6.0)


class TestAssociation:
    def test_stable_id_across_frames(self):
        tracker = ByteTracker()
        ids = {tracker.update([person(100 + i * 10)], at(i))[0].track_id for i in range(20)}
        assert ids == {1}

    def test_id_survives_low_confidence_occlusion(self):
        """THE reason we use ByteTrack: a person half-hidden behind a fence post
        drops to 0.2 confidence for a few frames, which is exactly when a border
        intrusion is most interesting."""
        tracker = ByteTracker()
        ids = []
        for i in range(30):
            conf = 0.15 if 12 <= i <= 16 else 0.9
            tracks = tracker.update([person(100 + i * 12, conf)], at(i))
            ids.extend(t.track_id for t in tracks)
        assert set(ids) == {1}

    def test_two_people_get_two_ids(self):
        tracker = ByteTracker()
        for i in range(10):
            tracks = tracker.update([person(100 + i * 5), person(700 + i * 5)], at(i))
        assert len({t.track_id for t in tracks}) == 2

    def test_classes_do_not_swap_identity(self):
        """A person box must not inherit a vehicle's track just because the
        boxes overlap."""
        tracker = ByteTracker()
        for i in range(10):
            tracks = tracker.update(
                [
                    Detection("person", 0.9, (100.0, 200.0, 140.0, 320.0), 0),
                    Detection("vehicle", 0.9, (105.0, 205.0, 300.0, 330.0), 2),
                ],
                at(i),
            )
        classes = {t.track_id: t.cls for t in tracks}
        assert set(classes.values()) == {"person", "vehicle"}

    def test_births_only_from_confident_detections(self):
        tracker = ByteTracker()
        assert tracker.update([person(100, 0.2)], at(0)) == []

    def test_no_detections_is_fine(self):
        tracker = ByteTracker()
        assert tracker.update([], at(0)) == []


class TestLifecycle:
    def test_confirmation_requires_min_hits(self):
        tracker = ByteTracker(TrackerConfig(min_hits=3))
        first = tracker.update([person(100)], at(0))[0]
        assert not first.is_confirmed
        tracker.update([person(110)], at(1))
        third = tracker.update([person(120)], at(2))[0]
        assert third.is_confirmed

    def test_track_expires_after_max_age(self):
        tracker = ByteTracker(TrackerConfig(max_age=5))
        for i in range(5):
            tracker.update([person(100 + i * 10)], at(i))
        for i in range(5, 20):
            tracker.update([], at(i))
        assert tracker.active_count == 0

    def test_close_expired_reports_dead_tracks_once(self):
        """This hook is what destroys non-matching face embeddings (P6)."""
        tracker = ByteTracker(TrackerConfig(max_age=3))
        for i in range(5):
            tracker.update([person(100 + i * 10)], at(i))
        closed_total = []
        for i in range(5, 20):
            tracker.update([], at(i))
            closed_total.extend(tracker.close_expired())
        assert len(closed_total) == 1
        assert closed_total[0].track_id == 1
        assert tracker.close_expired() == []  # drained, not repeated

    def test_history_is_capped(self):
        tracker = ByteTracker(TrackerConfig(history_len=10))
        for i in range(50):
            tracks = tracker.update([person(100 + i * 5)], at(i))
        assert len(tracks[0].history) <= 10

    def test_foot_point_is_bottom_centre(self):
        tracker = ByteTracker()
        track = tracker.update([Detection("person", 0.9, (100.0, 0.0, 200.0, 400.0), 0)], at(0))[0]
        fx, fy = track.foot_point
        assert fx == pytest.approx(150.0, abs=1.0)
        assert fy == pytest.approx(400.0, abs=1.0)


class TestPurity:
    def test_update_never_reads_the_clock(self):
        """Passing ts in is what lets us track an hour in a millisecond."""
        a = ByteTracker()
        b = ByteTracker()
        for i in range(20):
            ta = a.update([person(100 + i * 10)], at(i))
            tb = b.update([person(100 + i * 10)], at(i * 1000))  # absurd timestamps
        assert [t.track_id for t in ta] == [t.track_id for t in tb]
        assert [t.box for t in ta] == [t.box for t in tb]

    def test_reset(self):
        tracker = ByteTracker()
        tracker.update([person(100)], at(0))
        tracker.reset()
        assert tracker.active_count == 0


def test_config_from_mapping():
    cfg = TrackerConfig.from_mapping({"tracker": {"max_age": 99, "min_hits": 1}})
    assert cfg.max_age == 99
    assert cfg.min_hits == 1
    assert cfg.high_thresh == 0.50
