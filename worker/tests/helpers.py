"""Test factories, importable from any test module."""

from __future__ import annotations

from datetime import UTC, datetime

from drishti_worker.types import Detection, Track

T0 = datetime(2026, 9, 12, 22, 0, 0, tzinfo=UTC)


def make_track(
    *,
    track_id: int = 1,
    cls: str = "person",
    box: tuple[float, float, float, float] = (200.0, 200.0, 260.0, 400.0),
    hits: int = 10,
    age: int = 20,
    conf: float = 0.9,
    history: tuple[tuple[float, float], ...] | None = None,
    first_seen: datetime | None = None,
    last_seen: datetime | None = None,
) -> Track:
    fp = ((box[0] + box[2]) / 2.0, box[3])
    return Track(
        track_id=track_id,
        cls=cls,
        box=box,
        conf=conf,
        max_conf=conf,
        age_frames=age,
        hits=hits,
        time_since_update=0,
        first_seen=first_seen or T0,
        last_seen=last_seen or T0,
        history=history if history is not None else (fp, fp),
    )


def make_detection(
    cls: str = "person",
    conf: float = 0.9,
    box: tuple[float, float, float, float] = (200.0, 200.0, 260.0, 400.0),
) -> Detection:
    return Detection(cls=cls, conf=conf, box=box, cls_id=0)
