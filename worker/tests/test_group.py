"""Group movement analysis (§7.7). Pure functions, so property-based.

The sharp edge here is not the arithmetic, it is the false positive: a group
milling about must not read as a coordinated rush, or the operator learns to
ignore the signal and it may as well not exist (P3).
"""

from __future__ import annotations

import math

import pytest
from helpers import make_track
from hypothesis import given, settings
from hypothesis import strategies as st

from ibvap_worker.group import (
    centroid,
    classify_group_motion,
    spread,
    spread_series,
)

coords = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
points = st.tuples(coords, coords)


def line_of(xs: list[float], y: float = 0.0) -> list[tuple[float, float]]:
    return [(x, y) for x in xs]


class TestCentroid:
    def test_mean_of_a_square(self):
        assert centroid([(0, 0), (10, 0), (10, 10), (0, 10)]) == (5.0, 5.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            centroid([])

    @given(st.lists(points, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_centroid_lies_within_the_bounding_box(self, pts):
        cx, cy = centroid(pts)
        assert min(p[0] for p in pts) - 1e-6 <= cx <= max(p[0] for p in pts) + 1e-6
        assert min(p[1] for p in pts) - 1e-6 <= cy <= max(p[1] for p in pts) + 1e-6


class TestSpread:
    def test_identical_points_have_no_spread(self):
        assert spread([(5.0, 5.0)] * 4) == 0.0

    def test_single_point_is_not_an_error(self):
        # A one-person "group" is a legitimate input that simply never fires.
        assert spread([(1.0, 2.0)]) == 0.0
        assert spread([]) == 0.0

    def test_known_value(self):
        # Two points 10 apart: each sits 5 from the centroid, so the mean is 5.
        assert spread([(0.0, 0.0), (10.0, 0.0)]) == pytest.approx(5.0)

    @given(st.lists(points, min_size=2, max_size=15))
    @settings(max_examples=150)
    def test_spread_is_never_negative(self, pts):
        assert spread(pts) >= 0.0

    @given(st.lists(points, min_size=2, max_size=12), coords, coords)
    @settings(max_examples=150)
    def test_translation_invariant(self, pts, dx, dy):
        """Walking the whole group ten metres left changes nothing about how
        spread out it is. If this fails, the rule fires on camera pans."""
        moved = [(x + dx, y + dy) for x, y in pts]
        assert spread(moved) == pytest.approx(spread(pts), rel=1e-6, abs=1e-6)

    @given(st.lists(points, min_size=2, max_size=12), st.floats(0.1, 10.0))
    @settings(max_examples=150)
    def test_scales_linearly(self, pts, k):
        scaled = [(x * k, y * k) for x, y in pts]
        assert spread(scaled) == pytest.approx(spread(pts) * k, rel=1e-6, abs=1e-6)


class TestSpreadSeries:
    def _tracks_from(self, per_track_histories):
        return [
            make_track(track_id=i, history=tuple(h))
            for i, h in enumerate(per_track_histories, start=1)
        ]

    def test_oldest_to_newest(self):
        # Two tracks walking towards each other: spread must fall over time.
        a = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        b = [(100.0, 0.0), (90.0, 0.0), (80.0, 0.0)]
        series = spread_series(self._tracks_from([a, b]), window=3)
        assert series == pytest.approx([50.0, 40.0, 30.0])

    def test_tracks_without_enough_history_are_excluded(self):
        long_a = [(0.0, 0.0)] * 5
        long_b = [(100.0, 0.0)] * 5
        newborn = [(50.0, 0.0)]
        series = spread_series(self._tracks_from([long_a, long_b, newborn]), window=5)
        # Only the two long tracks count, so spread is a steady 50.
        assert series == pytest.approx([50.0] * 5)

    def test_fewer_than_two_usable_tracks_yields_nothing(self):
        assert spread_series(self._tracks_from([[(0.0, 0.0)] * 5]), window=5) == []
        assert spread_series([], window=5) == []

    def test_window_below_two_is_rejected(self):
        with pytest.raises(ValueError):
            spread_series(self._tracks_from([[(0.0, 0.0)] * 5] * 2), window=1)


class TestClassifyGroupMotion:
    def test_converging(self):
        series = [200.0, 170.0, 140.0, 110.0, 80.0]
        motion = classify_group_motion(series, members=4)
        assert motion is not None
        assert motion.code == "GROUP_CONVERGING"
        assert motion.members == 4
        assert motion.ratio == pytest.approx(0.4)

    def test_dispersing(self):
        series = [80.0, 110.0, 140.0, 170.0, 200.0]
        motion = classify_group_motion(series, members=5)
        assert motion is not None
        assert motion.code == "GROUP_DISPERSING"

    def test_milling_about_is_not_an_event(self):
        """The false positive that matters. A group drifting in and out within
        noise must produce nothing at all."""
        series = [150.0, 140.0, 155.0, 145.0, 150.0]
        assert classify_group_motion(series, members=4) is None

    def test_a_late_jitter_frame_cannot_invent_a_convergence(self):
        """Endpoints alone would call this a convergence. It ends at 80 having
        started at 200, but it was tighter mid-window and is now re-opening --
        that is not a group closing in."""
        series = [200.0, 90.0, 40.0, 60.0, 80.0]
        assert classify_group_motion(series, members=4) is None

    def test_tight_group_shuffling_is_ignored(self):
        """Small spreads make big ratios out of nothing. Three people standing
        in a huddle must not converge and disperse alternately forever."""
        series = [20.0, 15.0, 10.0, 8.0, 5.0]  # ratio 0.25, but tiny throughout
        assert classify_group_motion(series, members=3) is None

    def test_series_too_short(self):
        assert classify_group_motion([100.0], members=3) is None
        assert classify_group_motion([], members=3) is None

    def test_zero_starting_spread_is_not_a_division_error(self):
        assert classify_group_motion([0.0, 50.0], members=3) is None

    def test_nonsense_thresholds_are_rejected(self):
        with pytest.raises(ValueError):
            classify_group_motion([200.0, 80.0], members=3, converge_ratio=1.5)
        with pytest.raises(ValueError):
            classify_group_motion([80.0, 200.0], members=3, disperse_ratio=0.5)

    @given(
        st.lists(st.floats(min_value=1.0, max_value=5000.0), min_size=2, max_size=30),
        st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=200)
    def test_never_both_and_never_crashes(self, series, members):
        """Whatever the series, the answer is one code or none -- and the code
        always agrees with the direction the spread actually moved."""
        motion = classify_group_motion(series, members)
        if motion is None:
            return
        assert motion.code in ("GROUP_CONVERGING", "GROUP_DISPERSING")
        if motion.code == "GROUP_CONVERGING":
            assert motion.spread_to < motion.spread_from
        else:
            assert motion.spread_to > motion.spread_from
        assert math.isfinite(motion.ratio)
