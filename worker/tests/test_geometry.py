"""Geometry (§7.6). Property-based, because these are the sharpest edges."""

from __future__ import annotations

import math

import pytest
from helpers import make_track
from hypothesis import given, settings
from hypothesis import strategies as st

from drishti_worker.geometry import (
    crossing_direction,
    denormalise,
    dwell_seconds,
    iou,
    point_in_polygon,
    point_to_segment_distance,
    polygon_area,
    polygon_centroid,
    segments_intersect,
    speed_m_per_s,
    speed_px_per_s,
)
from drishti_worker.types import Calibration

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
coords = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


class TestPointInPolygon:
    def test_interior_and_exterior(self):
        assert point_in_polygon((5, 5), SQUARE)
        assert not point_in_polygon((15, 5), SQUARE)

    def test_edges_are_inclusive(self):
        # Defined behaviour. The alternative is a point that flickers in and out
        # under floating-point noise, which is an alert that fires twenty times.
        for pt in [(0, 5), (5, 0), (10, 5), (5, 10)]:
            assert point_in_polygon(pt, SQUARE), pt

    def test_vertices_are_inclusive(self):
        for pt in SQUARE:
            assert point_in_polygon(pt, SQUARE), pt

    def test_concave_polygon(self):
        # An L shape: the notch must be outside.
        poly = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
        assert point_in_polygon((2, 2), poly)
        assert point_in_polygon((8, 2), poly)
        assert not point_in_polygon((8, 8), poly)

    def test_winding_does_not_matter(self):
        assert point_in_polygon((5, 5), list(reversed(SQUARE)))

    @given(x=st.floats(0.5, 9.5), y=st.floats(0.5, 9.5), dx=coords, dy=coords)
    @settings(max_examples=200)
    def test_translation_invariance(self, x, y, dx, dy):
        """An interior point stays interior under translation."""
        moved = [(px + dx, py + dy) for px, py in SQUARE]
        assert point_in_polygon((x + dx, y + dy), moved)

    @given(theta=st.floats(0, 2 * math.pi))
    @settings(max_examples=100)
    def test_rotation_invariance(self, theta):
        centre = (5.0, 5.0)

        def rot(p):
            dx, dy = p[0] - centre[0], p[1] - centre[1]
            return (
                centre[0] + dx * math.cos(theta) - dy * math.sin(theta),
                centre[1] + dx * math.sin(theta) + dy * math.cos(theta),
            )

        assert point_in_polygon(centre, [rot(p) for p in SQUARE])

    def test_degenerate_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0, 0), [(0, 0), (1, 1)])
        with pytest.raises(ValueError):
            point_in_polygon((float("nan"), 0), SQUARE)


class TestSegments:
    def test_crossing_and_not(self):
        assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
        assert not segments_intersect((0, 0), (1, 1), (5, 5), (6, 6))

    def test_touching_counts(self):
        assert segments_intersect((0, 0), (5, 0), (5, 0), (5, 5))

    @given(
        a=st.tuples(coords, coords),
        b=st.tuples(coords, coords),
        c=st.tuples(coords, coords),
        d=st.tuples(coords, coords),
    )
    @settings(max_examples=300)
    def test_symmetry(self, a, b, c, d):
        """Symmetric in each pair, and between the pairs."""
        if a == b or c == d:
            return
        base = segments_intersect(a, b, c, d)
        assert segments_intersect(b, a, c, d) == base
        assert segments_intersect(a, b, d, c) == base
        assert segments_intersect(c, d, a, b) == base

    def test_zero_length_raises(self):
        with pytest.raises(ValueError):
            segments_intersect((0, 0), (0, 0), (1, 1), (2, 2))


class TestCrossingDirection:
    WIRE = ((0.0, 0.0), (10.0, 0.0))

    def test_opposite_directions_give_opposite_labels(self):
        assert crossing_direction((5, -1), (5, 1), self.WIRE) == "in"
        assert crossing_direction((5, 1), (5, -1), self.WIRE) == "out"

    def test_no_crossing(self):
        assert crossing_direction((5, 1), (5, 2), self.WIRE) is None

    def test_movement_along_the_wire_is_not_a_crossing(self):
        assert crossing_direction((1, 0), (9, 0), self.WIRE) is None

    @given(
        x=st.floats(0.5, 9.5),
        d=st.floats(0.1, 100.0),
    )
    @settings(max_examples=200)
    def test_reversal_is_always_opposite(self, x, d):
        forward = crossing_direction((x, -d), (x, d), self.WIRE)
        backward = crossing_direction((x, d), (x, -d), self.WIRE)
        assert {forward, backward} == {"in", "out"}

    def test_reversed_wire_flips_labels(self):
        reversed_wire = (self.WIRE[1], self.WIRE[0])
        assert crossing_direction((5, -1), (5, 1), reversed_wire) == "out"

    def test_zero_length_wire_raises(self):
        with pytest.raises(ValueError):
            crossing_direction((0, 0), (1, 1), ((5, 5), (5, 5)))


class TestMeasures:
    def test_area_and_centroid(self):
        assert polygon_area(SQUARE) == pytest.approx(100.0)
        assert polygon_centroid(SQUARE) == pytest.approx((5.0, 5.0))

    def test_area_is_winding_independent(self):
        assert polygon_area(list(reversed(SQUARE))) == pytest.approx(100.0)

    def test_point_to_segment(self):
        assert point_to_segment_distance((5, 3), (0, 0), (10, 0)) == pytest.approx(3.0)
        # Beyond the end: clamps to the endpoint, not the infinite line.
        assert point_to_segment_distance((13, 4), (0, 0), (10, 0)) == pytest.approx(5.0)

    def test_denormalise(self):
        assert denormalise([(0.5, 0.5)], 1280, 720) == [(640.0, 360.0)]
        with pytest.raises(ValueError):
            denormalise([(0.5, 0.5)], 0, 720)

    def test_iou(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    @given(a=st.floats(0, 100), b=st.floats(0, 100))
    @settings(max_examples=100)
    def test_iou_is_symmetric_and_bounded(self, a, b):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (a, b, a + 10, b + 10)
        value = iou(box1, box2)
        assert 0.0 <= value <= 1.0
        assert iou(box2, box1) == pytest.approx(value)


class TestTrackMeasures:
    def test_dwell_resets_on_exit(self):
        """Leaving and re-entering resets the clock — otherwise someone who
        walked past twice would eventually trip a loiter rule."""
        inside = (300.0, 300.0)
        outside = (900.0, 300.0)
        poly = [(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)]
        track = make_track(history=(inside,) * 10 + (outside,) + (inside,) * 3)
        assert dwell_seconds(track, poly, 6.0) == pytest.approx(0.5)  # only the last 3

    def test_speed(self):
        track = make_track(history=((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)))
        assert speed_px_per_s(track, 6.0) == pytest.approx(60.0)

    def test_speed_is_none_without_calibration(self):
        """A wrong speed is a wrong alert. Never guess a scale."""
        track = make_track(history=((0.0, 0.0), (10.0, 0.0)))
        assert speed_m_per_s(track, 6.0, None) is None
        assert speed_m_per_s(track, 6.0, Calibration()) is None

    def test_speed_with_calibration(self):
        track = make_track(history=((0.0, 400.0), (10.0, 400.0)))
        calib = Calibration(px_per_m_at_y=((400.0, 0.05),))
        assert speed_m_per_s(track, 6.0, calib) == pytest.approx(60.0 * 0.05)

    def test_bad_fps_raises(self):
        with pytest.raises(ValueError):
            speed_px_per_s(make_track(), 0.0)
