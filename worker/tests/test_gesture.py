"""Hand-signal recognition (§7.7). Pure geometry, so property-based.

The property that matters most is scale invariance. A pixel threshold would
silently work at exactly one distance from the camera — passing every
hand-written test built at that distance, and failing on the day.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ibvap_worker.gesture import (
    COCO_KEYPOINTS,
    GestureCandidate,
    GestureVoter,
    Keypoint,
    Pose,
    body_scale,
    classify_gesture,
    vote_gesture,
)


def build_pose(overrides: dict[str, tuple[float, float]], conf: float = 0.9) -> Pose:
    """A neutral standing figure, with named joints moved as requested.

    Coordinates are image-style: y grows downward. The default is a person
    ~100 px tall in the torso with both arms hanging straight down.
    """
    base: dict[str, tuple[float, float]] = {
        "nose": (100.0, 40.0),
        "left_eye": (95.0, 35.0),
        "right_eye": (105.0, 35.0),
        "left_ear": (90.0, 38.0),
        "right_ear": (110.0, 38.0),
        "left_shoulder": (80.0, 60.0),
        "right_shoulder": (120.0, 60.0),
        "left_elbow": (78.0, 110.0),
        "right_elbow": (122.0, 110.0),
        "left_wrist": (76.0, 160.0),
        "right_wrist": (124.0, 160.0),
        "left_hip": (85.0, 160.0),
        "right_hip": (115.0, 160.0),
        "left_knee": (85.0, 230.0),
        "right_knee": (115.0, 230.0),
        "left_ankle": (85.0, 300.0),
        "right_ankle": (115.0, 300.0),
    }
    base.update(overrides)
    return Pose(tuple(Keypoint(*base[name], conf) for name in COCO_KEYPOINTS))


HANDS_UP = {
    "left_wrist": (78.0, 10.0),
    "right_wrist": (122.0, 10.0),
    "left_elbow": (78.0, 35.0),
    "right_elbow": (122.0, 35.0),
}
ONE_ARM_UP = {"left_wrist": (78.0, 10.0), "left_elbow": (78.0, 35.0)}
POINTING_LEFT = {
    "left_shoulder": (80.0, 60.0),
    "left_elbow": (30.0, 60.0),
    "left_wrist": (-20.0, 60.0),
}


class TestPoseType:
    def test_wrong_keypoint_count_is_rejected(self):
        with pytest.raises(ValueError):
            Pose((Keypoint(0.0, 0.0, 1.0),) * 5)

    def test_low_confidence_keypoint_reads_as_absent(self):
        """An occluded wrist must make a gesture unknown, never invented."""
        pose = build_pose({}, conf=0.1)
        assert pose.get("left_wrist", 0.35) is None


class TestBodyScale:
    def test_prefers_torso(self):
        # Shoulders at y=60, hips at y=160 -> torso is 100 px.
        assert body_scale(build_pose({}), 0.35) == pytest.approx(100.0, abs=1.0)

    def test_falls_back_to_shoulder_width_when_hips_are_hidden(self):
        pose = build_pose({})
        points = list(pose.points)
        for name in ("left_hip", "right_hip"):
            idx = COCO_KEYPOINTS.index(name)
            points[idx] = Keypoint(points[idx].x, points[idx].y, 0.01)
        hidden_hips = Pose(tuple(points))
        assert body_scale(hidden_hips, 0.35) == pytest.approx(40.0)

    def test_none_without_shoulders(self):
        assert body_scale(build_pose({}, conf=0.05), 0.35) is None


class TestClassifyGesture:
    def test_hands_up(self):
        got = classify_gesture(build_pose(HANDS_UP))
        assert got is not None and got.code == "HANDS_UP"

    def test_one_arm_up_is_not_surrender(self):
        got = classify_gesture(build_pose(ONE_ARM_UP))
        assert got is not None and got.code == "ARM_RAISED"

    def test_arms_down_is_nothing(self):
        assert classify_gesture(build_pose({})) is None

    def test_pointing(self):
        got = classify_gesture(build_pose(POINTING_LEFT))
        assert got is not None and got.code == "POINTING"

    def test_a_bent_level_arm_is_not_pointing(self):
        """Hand on hip reaches sideways and sits level, but the elbow is far
        off the shoulder->wrist line. That is not a point."""
        bent = {
            "left_shoulder": (80.0, 60.0),
            "left_elbow": (20.0, 130.0),
            "left_wrist": (-20.0, 60.0),
        }
        got = classify_gesture(build_pose(bent))
        assert got is None or got.code != "POINTING"

    def test_a_hanging_arm_is_not_pointing(self):
        hanging = {
            "left_shoulder": (80.0, 60.0),
            "left_elbow": (50.0, 110.0),
            "left_wrist": (-20.0, 160.0),
        }
        got = classify_gesture(build_pose(hanging))
        assert got is None or got.code != "POINTING"

    def test_unknown_when_keypoints_are_not_trusted(self):
        assert classify_gesture(build_pose(HANDS_UP, conf=0.1)) is None

    @given(
        st.floats(min_value=0.25, max_value=6.0),
        st.floats(min_value=-3000.0, max_value=3000.0),
        st.floats(min_value=-3000.0, max_value=3000.0),
    )
    @settings(max_examples=150)
    def test_scale_and_translation_invariant(self, k, dx, dy):
        """The same gesture at any distance from the camera, anywhere in the
        frame, must classify the same. This is the property a pixel threshold
        would quietly fail."""
        for overrides, expected in (
            (HANDS_UP, "HANDS_UP"),
            (ONE_ARM_UP, "ARM_RAISED"),
            (POINTING_LEFT, "POINTING"),
        ):
            pose = build_pose(overrides)
            moved = Pose(tuple(Keypoint(p.x * k + dx, p.y * k + dy, p.conf) for p in pose.points))
            got = classify_gesture(moved)
            assert got is not None and got.code == expected, (expected, k, dx, dy)

    @given(st.floats(min_value=0.3, max_value=4.0))
    @settings(max_examples=100)
    def test_a_neutral_stance_never_fires_at_any_scale(self, k):
        pose = build_pose({})
        scaled = Pose(tuple(Keypoint(p.x * k, p.y * k, p.conf) for p in pose.points))
        assert classify_gesture(scaled) is None


class TestVoting:
    def test_needs_enough_agreement(self):
        three = [GestureCandidate("HANDS_UP", 0.9)] * 3
        assert vote_gesture(three, min_frames_agreed=4) is None
        assert vote_gesture([*three, GestureCandidate("HANDS_UP", 0.9)]) is not None

    def test_a_tie_has_not_settled(self):
        split = [GestureCandidate("HANDS_UP", 0.9)] * 4 + [GestureCandidate("POINTING", 0.9)] * 4
        assert vote_gesture(split, min_frames_agreed=4) is None

    def test_reports_frames_agreed(self):
        got = vote_gesture([GestureCandidate("POINTING", 0.8)] * 5, min_frames_agreed=4)
        assert got is not None
        assert got.detail["frames_agreed"] == 5.0

    def test_empty(self):
        assert vote_gesture([]) is None

    def test_rejects_nonsense_threshold(self):
        with pytest.raises(ValueError):
            vote_gesture([GestureCandidate("HANDS_UP", 1.0)], min_frames_agreed=0)


class TestGestureVoter:
    def test_a_limb_in_transit_never_settles(self):
        """An arm passes through 'raised' mid-stride. One frame must not fire —
        this is the difference between a signal and a walking person."""
        voter = GestureVoter(min_frames_agreed=4, window_frames=8)
        assert voter.add(GestureCandidate("ARM_RAISED", 0.9)) is None
        for _ in range(5):
            assert voter.add(None) is None

    def test_a_held_signal_settles(self):
        voter = GestureVoter(min_frames_agreed=4, window_frames=8)
        settled = None
        for _ in range(4):
            settled = voter.add(GestureCandidate("HANDS_UP", 0.9))
        assert settled is not None and settled.code == "HANDS_UP"

    def test_lowering_the_hands_un_settles(self):
        """A gesture is not a permanent fact about a person, unlike a plate."""
        voter = GestureVoter(min_frames_agreed=4, window_frames=8)
        for _ in range(4):
            voter.add(GestureCandidate("HANDS_UP", 0.9))
        for _ in range(8):
            last = voter.add(None)
        assert last is None

    def test_quiet_frames_count_against_the_window(self):
        voter = GestureVoter(min_frames_agreed=4, window_frames=8)
        for _ in range(3):
            voter.add(GestureCandidate("HANDS_UP", 0.9))
            voter.add(None)
        # Three raised frames among six is not agreement.
        assert voter.add(None) is None

    def test_reset(self):
        voter = GestureVoter(min_frames_agreed=2, window_frames=8)
        voter.add(GestureCandidate("HANDS_UP", 0.9))
        voter.reset()
        assert voter.add(GestureCandidate("HANDS_UP", 0.9)) is None


def test_no_gesture_code_collides_with_the_sentinel():
    """`GestureVoter` uses the code "NONE" internally to mean "nothing this
    frame". If a real gesture were ever named that, it would be swallowed."""
    from ibvap_worker.gesture import GESTURE_CODES

    assert "NONE" not in GESTURE_CODES
    assert all(math.isfinite(len(code)) for code in GESTURE_CODES)
