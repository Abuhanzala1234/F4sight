"""Geometry primitives (BUILD_SPEC §7.6).

Pure functions, no dependencies beyond the standard library. This is deliberate:
these are the sharpest edge cases in the system and they get property-based
tests, so they must be importable and runnable without numpy, OpenCV or a GPU.

Two conventions that everything else depends on:

* **Edge-inclusive containment.** A point exactly on a polygon edge is INSIDE.
  The alternative is a point that flickers in and out as floating point noise
  moves it across the boundary, which is an alert that fires twenty times.
  Deterministic and slightly wrong beats non-deterministic and slightly right.

* **Foot-points, not centroids.** Callers pass ``track.foot_point``. A person's
  centroid is a metre off the ground and crosses a tripwire before they do.

Degenerate input raises ``ValueError``. It never returns a silently wrong
answer — CLAUDE.md: no silent failures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .types import Calibration, Point, Track

__all__ = [
    "crossing_direction",
    "denormalise",
    "dwell_seconds",
    "iou",
    "normalise",
    "perpendicular_travel",
    "point_in_polygon",
    "point_to_segment_distance",
    "polygon_area",
    "polygon_centroid",
    "segments_intersect",
    "speed_m_per_s",
    "speed_px_per_s",
]

_EPS = 1e-9


def _validate_point(pt: Point, what: str = "point") -> None:
    if len(pt) != 2:
        raise ValueError(f"{what} must be (x, y), got {pt!r}")
    x, y = pt
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{what} has non-finite coordinate: {pt!r}")


def _validate_polygon(poly: Sequence[Point], min_points: int = 3) -> None:
    if len(poly) < min_points:
        raise ValueError(f"polygon needs >= {min_points} points, got {len(poly)}")
    for p in poly:
        _validate_point(p, "polygon vertex")


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


def _on_segment(pt: Point, a: Point, b: Point) -> bool:
    """True if ``pt`` lies on segment a-b (within epsilon)."""
    px, py = pt
    ax, ay = a
    bx, by = b
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > _EPS * max(1.0, abs(bx - ax) + abs(by - ay)):
        return False
    # Collinear: check it is within the bounding box of the segment.
    return (
        min(ax, bx) - _EPS <= px <= max(ax, bx) + _EPS
        and min(ay, by) - _EPS <= py <= max(ay, by) + _EPS
    )


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    """Ray casting, edge-inclusive.

    A point on any edge or vertex returns True. Interior points return True.
    The polygon may be convex or concave, wound either way.
    """
    _validate_point(pt)
    _validate_polygon(poly)

    x, y = pt
    n = len(poly)

    # Edge-inclusive: settle boundary cases before the parity test, because the
    # parity test's answer on a boundary depends on rounding.
    for i in range(n):
        if _on_segment(pt, poly[i], poly[(i + 1) % n]):
            return True

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Half-open rule on y (yi <= y < yj or yj <= y < yi) so that a ray
        # passing exactly through a vertex is counted once, not twice.
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Crossings
# ---------------------------------------------------------------------------


def _orientation(a: Point, b: Point, c: Point) -> int:
    """-1 clockwise, +1 counter-clockwise, 0 collinear."""
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < _EPS:
        return 0
    return 1 if val < 0 else -1


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 intersects segment q1-q2, touching included.

    Symmetric in (p1,p2) and in (q1,q2), and symmetric between the two pairs —
    all three properties are asserted by the property tests.
    """
    for p in (p1, p2, q1, q2):
        _validate_point(p)
    if p1 == p2 or q1 == q2:
        raise ValueError("zero-length segment")

    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True
    # Collinear touching cases.
    if o1 == 0 and _on_segment(q1, p1, p2):
        return True
    if o2 == 0 and _on_segment(q2, p1, p2):
        return True
    if o3 == 0 and _on_segment(p1, q1, q2):
        return True
    return bool(o4 == 0 and _on_segment(p2, q1, q2))


def crossing_direction(
    prev: Point, cur: Point, wire: tuple[Point, Point]
) -> str | None:
    """Which way a track crossed a tripwire.

    Returns ``'in'``, ``'out'``, or ``None`` if the movement did not cross.

    Convention: walking from the wire's first point toward its second, ``'in'``
    is a crossing from the LEFT side to the RIGHT side (negative to positive
    side of the wire's normal). Drawing a wire in the reverse order flips the
    labels — which is why the zone editor shows an arrow.
    """
    _validate_point(prev, "prev")
    _validate_point(cur, "cur")
    a, b = wire
    _validate_point(a, "wire start")
    _validate_point(b, "wire end")
    if a == b:
        raise ValueError("zero-length tripwire")
    if prev == cur:
        return None
    if not segments_intersect(prev, cur, a, b):
        return None

    def side(p: Point) -> float:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    s_prev, s_cur = side(prev), side(cur)
    if s_prev == 0.0 and s_cur == 0.0:
        return None  # movement along the wire is not a crossing
    # Use the non-zero endpoint when one lies exactly on the wire.
    if s_prev == 0.0:
        return "in" if s_cur > 0 else "out"
    if s_cur == 0.0:
        return "out" if s_prev > 0 else "in"
    if s_prev < 0 < s_cur:
        return "in"
    if s_cur < 0 < s_prev:
        return "out"
    return None


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def perpendicular_travel(prev: Point, cur: Point, wire: tuple[Point, Point]) -> float:
    """Distance travelled PERPENDICULAR to a wire, in pixels.

    This is the right quantity for jitter suppression on a tripwire. Total
    motion is the wrong one: a track walking *along* a wire moves a great deal
    in total while never approaching it, and a track stepping decisively across
    at 6 fps moves only ~10 px in total. Measuring the normal component
    separates the two.
    """
    a, b = wire
    _validate_point(prev, "prev")
    _validate_point(cur, "cur")
    _validate_point(a, "wire start")
    _validate_point(b, "wire end")
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < _EPS:
        raise ValueError("zero-length tripwire")

    def side(p: Point) -> float:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    return abs(side(cur) - side(prev)) / length


def denormalise(poly: Sequence[Point], w: int, h: int) -> list[Point]:
    """Normalised (0..1) polygon -> pixels.

    Zones are stored normalised (§6.2) so that changing a camera's resolution
    does not invalidate every polygon an operator drew.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"bad frame size: {w}x{h}")
    out: list[Point] = []
    for p in poly:
        _validate_point(p, "normalised vertex")
        out.append((p[0] * w, p[1] * h))
    return out


def normalise(poly: Sequence[Point], w: int, h: int) -> list[Point]:
    """Pixels -> normalised (0..1)."""
    if w <= 0 or h <= 0:
        raise ValueError(f"bad frame size: {w}x{h}")
    return [(p[0] / w, p[1] / h) for p in poly]


def polygon_area(poly: Sequence[Point]) -> float:
    """Absolute shoelace area. Winding-independent."""
    _validate_polygon(poly)
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_centroid(poly: Sequence[Point]) -> Point:
    """Area centroid; falls back to the vertex mean for degenerate polygons."""
    _validate_polygon(poly)
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < _EPS:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    a *= 0.5
    return (cx / (6.0 * a), cy / (6.0 * a))


def point_to_segment_distance(pt: Point, a: Point, b: Point) -> float:
    """Shortest distance from a point to a segment. Used by PERIMETER_APPROACH."""
    _validate_point(pt)
    _validate_point(a, "segment start")
    _validate_point(b, "segment end")
    ax, ay = a
    bx, by = b
    px, py = pt
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < _EPS:
        raise ValueError("zero-length segment")
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection over union of two xyxy boxes. 0.0 when either is degenerate."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > _EPS else 0.0


# ---------------------------------------------------------------------------
# Track-derived measures
# ---------------------------------------------------------------------------


def dwell_seconds(track: Track, poly: Sequence[Point], fps: float) -> float:
    """Seconds the track's foot-point has been continuously inside ``poly``.

    Counts backward from the newest history point and stops at the first sample
    outside, so leaving and re-entering resets the clock. A loiter timer that
    accumulated across exits would fire on someone who walked past twice.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    _validate_polygon(poly)
    if not track.history:
        return 0.0
    consecutive = 0
    for pt in reversed(track.history):
        if not point_in_polygon(pt, poly):
            break
        consecutive += 1
    return consecutive / fps


def speed_px_per_s(track: Track, fps: float, window: int = 5) -> float:
    """Recent speed in pixels per second, averaged over the last ``window`` steps."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    hist = track.history
    if len(hist) < 2:
        return 0.0
    n = min(window + 1, len(hist))
    segment = hist[-n:]
    dist = sum(
        math.hypot(segment[i + 1][0] - segment[i][0], segment[i + 1][1] - segment[i][1])
        for i in range(len(segment) - 1)
    )
    return dist * fps / (len(segment) - 1)


def speed_m_per_s(
    track: Track, fps: float, calib: Calibration | None, window: int = 5
) -> float | None:
    """Real-world speed, or ``None`` when the camera is not calibrated.

    We return None rather than assuming a scale. An invented metres-per-second
    figure produces a confident, wrong alert, and P3 says a wrong alert is worse
    than a missing number.
    """
    if calib is None or not calib.px_per_m_at_y:
        return None
    px_s = speed_px_per_s(track, fps, window)
    y = track.foot_point[1]
    # Nearest calibration sample by image row; linear in y is good enough for
    # the shallow angles CCTV actually uses.
    nearest = min(calib.px_per_m_at_y, key=lambda row: abs(row[0] - y))
    metres_per_px = nearest[1]
    if metres_per_px <= 0:
        raise ValueError(f"bad calibration scale at y={y}: {metres_per_px}")
    return px_s * metres_per_px
