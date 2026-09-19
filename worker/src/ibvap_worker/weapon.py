"""Weapon detection on tracked people (BUILD_SPEC §7.7, weapons).

Second stage, crop-fed, same shape as gestures (§7.7) and ANPR (§7.9): it runs
on crops of person tracks, so it costs nothing when disabled and nothing when
nobody is being tracked, and the primary detector's hot path is untouched.

**This is the highest false-positive risk in the whole system, and the design
is shaped around that.** At CCTV distance a phone, an umbrella, a walking stick
and a farm tool all occupy the same few dozen pixels as a pistol. An armed-person
alert that cries wolf is worse than no alert at all, because the operator learns
to dismiss it — and the one time it is real, they dismiss that too (P3).

Three things hold the line:

1. A high confidence floor, separate from and stricter than the detector's.
2. Multi-frame voting, as with plates and gestures: a weapon that appears for a
   single frame is a misread, not an arrest.
3. Voting on ARMED, not on *which* weapon. Gun-vs-knife plurality would tie and
   never settle on a person the model keeps re-classifying between the two —
   and somebody the model cannot decide is holding a gun or a knife is still,
   unambiguously, holding something. The weapon *type* is reported as detail,
   never used as the gate.

Pure module: no I/O, no model, no numpy. The ONNX backend is in
detect/onnx_weapon.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "WEAPON_CLASSES",
    "WeaponCandidate",
    "WeaponConfig",
    "WeaponSettled",
    "WeaponVoter",
    "vote_weapon",
]

#: What the bundled model emits. Kept here rather than read from the ONNX
#: metadata so a swapped-in model with different class ids fails loudly at
#: config time instead of silently relabelling knives as guns.
WEAPON_CLASSES: tuple[str, ...] = ("guns", "knife")


@dataclass(frozen=True, slots=True)
class WeaponConfig:
    """The ``weapons:`` block of the merged config (config/weapons.yaml)."""

    enabled: bool = False
    weights: str = "models/weapon/weapon-yolov8.onnx"
    input_size: tuple[int, int] = (640, 640)
    classes: tuple[str, ...] = ("person",)
    #: Measured, not guessed -- see config/weapons.yaml for the curve. At 0.55
    #: this flagged 2 of 17 real pedestrians as armed.
    min_conf: float = 0.75
    nms_iou: float = 0.45
    every_n_frames: int = 3
    max_tracks_per_frame: int = 2
    min_frames_agreed: int = 3
    window_frames: int = 8
    #: Cheap "has this crop changed" gate in front of the model (activity.py).
    activity_gate: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> WeaponConfig:
        block = dict(cfg.get("weapons", cfg))
        voting = dict(block.get("voting", {}) or {})
        size = list(block.get("input_size", (640, 640)))
        return cls(
            enabled=bool(block.get("enabled", False)),
            weights=str(block.get("weights", "models/weapon/weapon-yolov8.onnx")),
            input_size=(int(size[0]), int(size[1])),
            classes=tuple(block.get("classes", ("person",))),
            min_conf=float(block.get("min_conf", 0.75)),
            nms_iou=float(block.get("nms_iou", 0.45)),
            every_n_frames=max(1, int(block.get("every_n_frames", 3))),
            max_tracks_per_frame=int(block.get("max_tracks_per_frame", 2)),
            min_frames_agreed=int(voting.get("min_frames_agreed", 3)),
            window_frames=int(voting.get("window_frames", 8)),
            activity_gate=dict(block.get("activity_gate", {}) or {}),
        )


@dataclass(frozen=True, slots=True)
class WeaponCandidate:
    """One frame's strongest weapon detection inside one person's crop."""

    cls: str
    conf: float


@dataclass(frozen=True, slots=True)
class WeaponSettled:
    """A weapon confirmed across enough frames to be worth an operator's time."""

    cls: str  # most-seen weapon type, for the operator -- never the gate
    conf: float  # mean confidence of the agreeing frames
    frames_agreed: int
    frames_seen: int


def vote_weapon(
    candidates: Sequence[WeaponCandidate | None], *, min_frames_agreed: int = 3
) -> WeaponSettled | None:
    """Settle ARMED / not-armed over a window of frames.

    ``None`` entries are "this frame saw no weapon" and must be passed in, not
    skipped: they are evidence too, and they are what lets a person who put
    something down stop being flagged.
    """
    if min_frames_agreed < 1:
        raise ValueError(f"min_frames_agreed must be positive, got {min_frames_agreed}")

    seen = [c for c in candidates if c is not None]
    if len(seen) < min_frames_agreed:
        return None

    # Which type to SHOW. Ties are broken by total confidence rather than left
    # unresolved, because the alert has to say something and "armed" is the
    # part that matters -- see the module docstring.
    by_class: dict[str, list[float]] = {}
    for c in seen:
        by_class.setdefault(c.cls, []).append(c.conf)
    best = max(by_class.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))

    return WeaponSettled(
        cls=best[0],
        conf=round(sum(c.conf for c in seen) / len(seen), 3),
        frames_agreed=len(seen),
        frames_seen=len(candidates),
    )


@dataclass
class WeaponVoter:
    """Rolling per-track window (mirrors GestureVoter).

    Never latches: a person who set something down stops being armed, so this
    reports what the recent window says every time it is asked.
    """

    min_frames_agreed: int = 3
    window_frames: int = 8
    _candidates: list[WeaponCandidate | None] = field(default_factory=list, repr=False)

    def add(self, candidate: WeaponCandidate | None) -> WeaponSettled | None:
        self._candidates.append(candidate)
        if len(self._candidates) > self.window_frames:
            self._candidates = self._candidates[-self.window_frames :]
        return vote_weapon(self._candidates, min_frames_agreed=self.min_frames_agreed)

    def reset(self) -> None:
        self._candidates.clear()
