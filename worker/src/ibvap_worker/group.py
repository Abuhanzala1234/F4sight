"""Group movement analysis (BUILD_SPEC §7.7, group behaviour rules).

Two people walking towards the same gate is a queue. Six people closing on one
stretch of fence from six directions is a rush — and the coordinated case is
exactly the one the per-track rules cannot see. Every individual track can stay
outside every zone, never cross a wire and never loiter, while the *formation*
those tracks make is the whole event.

``CROWD_FORMING`` in rules.py already answers "how many people are standing in
this zone". This module answers a different question — *what is the group
doing* — by measuring how the spread of a set of tracks changes over a short
window:

* spread shrinking → ``GROUP_CONVERGING``: several people closing on one point.
* spread growing → ``GROUP_DISPERSING``: the scatter that follows being seen.

Both are camera-wide rather than zone-bound on purpose. People converging on a
fence approach it from outside every drawn zone, by definition; requiring them
to already be inside one would mean the signal only ever fires after the thing
it is meant to give warning of.

Pure module: no I/O, no numpy, no clock, everything arrives through arguments.
That is what lets it be property-tested the same way geometry and risk are.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .types import Point, Track

__all__ = [
    "GroupMotion",
    "centroid",
    "classify_group_motion",
    "spread",
    "spread_series",
]


def centroid(points: Sequence[Point]) -> Point:
    """Arithmetic mean of the points. Raises on an empty sequence."""
    if not points:
        raise ValueError("centroid of no points is undefined")
    n = float(len(points))
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def spread(points: Sequence[Point]) -> float:
    """Mean distance from the centroid — how scattered this group is, in px.

    Mean-distance-to-centroid rather than mean pairwise distance: it is O(n)
    instead of O(n²), and it stays interpretable — "these people are on average
    120 px from their common centre" is a sentence an operator can check against
    the picture. Fewer than two points have no spread, which is 0.0, not an
    error: a one-person "group" is a legitimate input that simply never
    triggers anything.
    """
    if len(points) < 2:
        return 0.0
    cx, cy = centroid(points)
    return sum(math.hypot(p[0] - cx, p[1] - cy) for p in points) / float(len(points))


def spread_series(tracks: Sequence[Track], window: int) -> list[float]:
    """Group spread at each of the last ``window`` frames, oldest → newest.

    Only tracks with at least ``window`` history points take part. That is a
    deliberate exclusion rather than a best-effort average: a track that was
    born three frames ago has no opinion about what the group was doing two
    seconds ago, and letting it join mid-window would move the centroid for
    reasons that have nothing to do with anybody walking anywhere. A group
    whose members keep appearing and disappearing simply produces no series,
    which is the honest answer.

    Returns ``[]`` when fewer than two tracks qualify.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2 frames, got {window}")
    usable = [t for t in tracks if len(t.history) >= window]
    if len(usable) < 2:
        return []
    # history is oldest -> newest, so index -1-k is "k frames ago".
    return [spread([t.history[-1 - k] for t in usable]) for k in range(window - 1, -1, -1)]


@dataclass(frozen=True, slots=True)
class GroupMotion:
    """What a spread series says the group is doing. ``members`` is how many
    tracks actually contributed, which is never more than were offered."""

    code: str  # GROUP_CONVERGING | GROUP_DISPERSING
    spread_from: float
    spread_to: float
    ratio: float
    members: int


def classify_group_motion(
    series: Sequence[float],
    members: int,
    *,
    converge_ratio: float = 0.70,
    disperse_ratio: float = 1.40,
    min_spread_px: float = 60.0,
) -> GroupMotion | None:
    """Decide whether a spread series is a convergence, a dispersal, or noise.

    Two conditions must both hold, and the second is what keeps this honest:

    1. The endpoints moved far enough — spread fell to ``converge_ratio`` of
       where it started, or rose to ``disperse_ratio`` of it.
    2. The series *ended* at its own extreme (lowest point for a convergence,
       highest for a dispersal). Endpoints alone would let one jittery frame at
       either end invent an event out of a group that milled about and went
       nowhere. Full strict monotonicity — what PerimeterApproachRule demands
       of a single track — is too much to ask of six independent people, none
       of whom walk smoothly; "ended at the extreme" is the achievable middle.

    ``min_spread_px`` guards the degenerate end. A group already standing
    shoulder to shoulder has a tiny spread, and tiny numbers make large ratios
    out of nothing: three people shuffling within a metre of each other would
    otherwise "converge" and "disperse" alternately, forever.
    """
    if converge_ratio >= 1.0 or disperse_ratio <= 1.0:
        raise ValueError(
            f"converge_ratio must be < 1 and disperse_ratio > 1, "
            f"got {converge_ratio} and {disperse_ratio}"
        )
    if len(series) < 2:
        return None

    first, last = series[0], series[-1]
    if first <= 0.0:
        return None

    ratio = last / first

    # Converging: started meaningfully spread out, ended tight, and tightest
    # right now.
    if first >= min_spread_px and ratio <= converge_ratio and last <= min(series):
        return GroupMotion("GROUP_CONVERGING", first, last, ratio, members)

    # Dispersing: ended meaningfully spread out, having grown, and widest now.
    if last >= min_spread_px and ratio >= disperse_ratio and last >= max(series):
        return GroupMotion("GROUP_DISPERSING", first, last, ratio, members)

    return None
