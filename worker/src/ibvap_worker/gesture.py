"""Hand-signal recognition from body keypoints (BUILD_SPEC §7.7, gestures).

A person on a fence line is not only a position, they are a posture. Somebody
standing with both hands above their head is doing something categorically
different from somebody waving one arm at a treeline — the first is complying,
the second is signalling to a person the camera cannot see. Neither is visible
to a rule that only knows where a bounding box is.

Three signals, chosen because each is unambiguous at CCTV distance and each
means something different operationally:

* ``HANDS_UP`` — both wrists above both shoulders. Surrender or compliance.
  Weighted *negatively*: a person showing their hands is de-escalating, and the
  score should say so (P3).
* ``ARM_RAISED`` — one arm up. Signalling across the line to someone else.
* ``POINTING`` — one arm straight and level. Directing others to a place.

Everything here is scale- and translation-invariant: all distances are measured
in units of the subject's own torso, never pixels. A gesture has to classify
identically whether the person is ten metres from the camera or two, and a
pixel threshold cannot do that — it would silently only ever work at one
distance, which is the kind of bug that passes every test and fails the demo.

Pure module: no I/O, no numpy, no model. The ONNX pose backend lives in
pose.py and hands this module plain numbers, which is what keeps the
interesting half testable without a GPU or a download.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .geometry import point_to_segment_distance
from .types import Point

__all__ = [
    "COCO_KEYPOINTS",
    "GESTURE_CODES",
    "GestureCandidate",
    "GestureConfig",
    "GestureVoter",
    "Keypoint",
    "Pose",
    "classify_gesture",
    "vote_gesture",
]

#: COCO-17 keypoint order, which is what every YOLO-pose export emits.
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

GESTURE_CODES: tuple[str, ...] = ("HANDS_UP", "ARM_RAISED", "POINTING")


@dataclass(frozen=True, slots=True)
class GestureConfig:
    """The ``gestures:`` block of the merged config (config/gestures.yaml)."""

    enabled: bool = False
    weights: str = "models/pose/yolo11n-pose.onnx"
    input_size: tuple[int, int] = (640, 640)
    classes: tuple[str, ...] = ("person",)
    min_person_conf: float = 0.40
    min_keypoint_conf: float = 0.35
    every_n_frames: int = 3
    max_tracks_per_frame: int = 2
    min_frames_agreed: int = 4
    window_frames: int = 8
    raise_margin: float = 0.15
    level_tolerance: float = 0.25
    extend_ratio: float = 0.85
    straight_tolerance: float = 0.30
    #: Cheap "has this crop changed" gate in front of the model (activity.py).
    #: Kept as a raw mapping so activity.py owns its own schema.
    activity_gate: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> GestureConfig:
        block = dict(cfg.get("gestures", cfg))
        voting = dict(block.get("voting", {}) or {})
        thresholds = dict(block.get("thresholds", {}) or {})
        size = list(block.get("input_size", (640, 640)))
        return cls(
            enabled=bool(block.get("enabled", False)),
            weights=str(block.get("weights", "models/pose/yolo11n-pose.onnx")),
            input_size=(int(size[0]), int(size[1])),
            classes=tuple(block.get("classes", ("person",))),
            min_person_conf=float(block.get("min_person_conf", 0.40)),
            min_keypoint_conf=float(block.get("min_keypoint_conf", 0.35)),
            every_n_frames=max(1, int(block.get("every_n_frames", 3))),
            max_tracks_per_frame=int(block.get("max_tracks_per_frame", 2)),
            min_frames_agreed=int(voting.get("min_frames_agreed", 4)),
            window_frames=int(voting.get("window_frames", 8)),
            raise_margin=float(thresholds.get("raise_margin", 0.15)),
            level_tolerance=float(thresholds.get("level_tolerance", 0.25)),
            extend_ratio=float(thresholds.get("extend_ratio", 0.85)),
            straight_tolerance=float(thresholds.get("straight_tolerance", 0.30)),
            activity_gate=dict(block.get("activity_gate", {}) or {}),
        )


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: float
    y: float
    conf: float


@dataclass(frozen=True, slots=True)
class Pose:
    """One person's 17 COCO keypoints, in ORIGINAL frame coordinates.

    Same invariant as every other geometric type here (§7.4): model-space
    coordinates never escape the backend that produced them.
    """

    points: tuple[Keypoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(COCO_KEYPOINTS):
            raise ValueError(
                f"expected {len(COCO_KEYPOINTS)} COCO keypoints, got {len(self.points)}"
            )

    def get(self, name: str, min_conf: float) -> Point | None:
        """A keypoint, or None when the model was not confident about it.

        Returning None rather than a low-confidence guess is the whole point:
        an occluded wrist must make a gesture *unknown*, never make it up.
        """
        kp = self.points[_INDEX[name]]
        return None if kp.conf < min_conf else (kp.x, kp.y)


@dataclass(frozen=True, slots=True)
class GestureCandidate:
    """One frame's opinion about what a person is doing."""

    code: str
    conf: float
    detail: dict[str, float] = field(default_factory=dict)


def _midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def body_scale(pose: Pose, min_conf: float) -> float | None:
    """The subject's own size, in pixels, for normalising every threshold.

    Torso length (shoulders to hips) first, because it is the most stable
    dimension of a standing human and barely changes as they turn. Shoulder
    width is the fallback for the very common CCTV case where the hips are
    behind a wall, a vehicle or the bottom of the frame — it is a weaker proxy
    because it foreshortens when someone turns side-on, hence second choice.

    Returns None when neither is measurable, which makes every gesture
    unknown rather than wrong.
    """
    ls = pose.get("left_shoulder", min_conf)
    rs = pose.get("right_shoulder", min_conf)
    if ls is None or rs is None:
        return None

    lh = pose.get("left_hip", min_conf)
    rh = pose.get("right_hip", min_conf)
    if lh is not None and rh is not None:
        shoulders, hips = _midpoint(ls, rs), _midpoint(lh, rh)
        torso = math.hypot(hips[0] - shoulders[0], hips[1] - shoulders[1])
        if torso > 1e-6:
            return torso

    width = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
    return width if width > 1e-6 else None


def _arm(pose: Pose, side: str, min_conf: float) -> tuple[Point, Point, Point] | None:
    """(shoulder, elbow, wrist) for one side, or None if any part is unsure."""
    shoulder = pose.get(f"{side}_shoulder", min_conf)
    elbow = pose.get(f"{side}_elbow", min_conf)
    wrist = pose.get(f"{side}_wrist", min_conf)
    if shoulder is None or elbow is None or wrist is None:
        return None
    return shoulder, elbow, wrist


def classify_gesture(
    pose: Pose,
    *,
    min_keypoint_conf: float = 0.35,
    raise_margin: float = 0.15,
    level_tolerance: float = 0.25,
    extend_ratio: float = 0.85,
    straight_tolerance: float = 0.30,
) -> GestureCandidate | None:
    """Classify one frame's pose into a hand signal, or None for "nothing".

    Thresholds are all *multiples of torso length*, never pixels — see the
    module docstring. Ordering matters: HANDS_UP is tested before ARM_RAISED
    because both arms up satisfies the weaker one-arm condition too, and the
    two mean very different things.

    Image coordinates, so "above" means a smaller y. Getting that backwards
    produces a classifier that detects surrender exactly when someone's hands
    are by their knees.
    """
    scale = body_scale(pose, min_keypoint_conf)
    if scale is None:
        return None

    left = _arm(pose, "left", min_keypoint_conf)
    right = _arm(pose, "right", min_keypoint_conf)
    if left is None and right is None:
        return None

    margin = raise_margin * scale

    def raised(arm: tuple[Point, Point, Point] | None) -> float | None:
        """How far the wrist is above its own shoulder, in px. None if unsure."""
        if arm is None:
            return None
        shoulder, _elbow, wrist = arm
        return shoulder[1] - wrist[1]

    left_lift, right_lift = raised(left), raised(right)

    # --- HANDS_UP: both wrists clearly above both shoulders.
    if (
        left_lift is not None
        and right_lift is not None
        and left_lift > margin
        and right_lift > margin
    ):
        lift = min(left_lift, right_lift) / scale
        return GestureCandidate(
            "HANDS_UP",
            conf=min(1.0, lift / max(raise_margin, 1e-6) / 4.0),
            detail={"lift_torsos": round(lift, 3)},
        )

    # --- ARM_RAISED: exactly one wrist above its shoulder.
    lifts = [(side, v) for side, v in (("left", left_lift), ("right", right_lift)) if v is not None]
    up = [(side, v) for side, v in lifts if v > margin]
    if len(up) == 1:
        side, value = up[0]
        lift = value / scale
        return GestureCandidate(
            "ARM_RAISED",
            conf=min(1.0, lift / max(raise_margin, 1e-6) / 4.0),
            detail={"lift_torsos": round(lift, 3), "side_is_left": float(side == "left")},
        )

    # --- POINTING: an arm held straight and level, reaching away from the body.
    for side, arm in (("left", left), ("right", right)):
        if arm is None:
            continue
        shoulder, elbow, wrist = arm
        reach = abs(wrist[0] - shoulder[0])
        drop = abs(wrist[1] - shoulder[1])
        if reach < extend_ratio * scale:
            continue  # hand is not out away from the body
        if drop > level_tolerance * scale:
            continue  # not level: this is a hanging or a raised arm, not a point
        # Straightness, reusing the tested primitive: a bent arm puts the elbow
        # well off the shoulder->wrist line, and a bent arm is not a point.
        if point_to_segment_distance(elbow, shoulder, wrist) > straight_tolerance * scale:
            continue
        return GestureCandidate(
            "POINTING",
            conf=min(1.0, reach / scale / max(extend_ratio, 1e-6) / 1.5),
            detail={
                "reach_torsos": round(reach / scale, 3),
                "side_is_left": float(side == "left"),
            },
        )

    return None


def vote_gesture(
    candidates: Sequence[GestureCandidate], *, min_frames_agreed: int = 4
) -> GestureCandidate | None:
    """Settle on a gesture only once enough recent frames agree.

    The same discipline ANPR applies to plate text, and for the same reason: a
    single frame is an opinion, not a fact. An arm passes through "raised" in
    the middle of an ordinary stride, and a classifier that fires on one frame
    reports every walking person as signalling. Requiring agreement across
    frames is what separates a held, deliberate signal from a limb in transit.

    The winner is the most common code among the candidates, and it must both
    reach ``min_frames_agreed`` and be a strict plurality — a window split
    evenly between two gestures has not settled on either.
    """
    if min_frames_agreed < 1:
        raise ValueError(f"min_frames_agreed must be positive, got {min_frames_agreed}")
    if not candidates:
        return None

    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.code] = counts.get(candidate.code, 0) + 1

    best_code, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count < min_frames_agreed:
        return None
    if any(code != best_code and count >= best_count for code, count in counts.items()):
        return None

    agreeing = [c for c in candidates if c.code == best_code]
    return GestureCandidate(
        best_code,
        conf=round(sum(c.conf for c in agreeing) / len(agreeing), 3),
        detail={**agreeing[-1].detail, "frames_agreed": float(best_count)},
    )


@dataclass
class GestureVoter:
    """Accumulates per-frame candidates for one track (mirrors PlateVoter).

    Stateful, but deliberately dumb: it holds a rolling window and defers every
    decision to ``vote_gesture`` above, so the logic stays testable without
    constructing a track.

    Unlike a plate, a gesture is not a permanent fact about the object — a
    person lowers their hands. So this never latches: it reports what the
    recent window says, every time it is asked.
    """

    min_frames_agreed: int = 4
    window_frames: int = 8
    _candidates: list[GestureCandidate] = field(default_factory=list, repr=False)

    def add(self, candidate: GestureCandidate | None) -> GestureCandidate | None:
        """Record one frame's opinion (None means "no gesture this frame") and
        return the settled gesture, if the window now agrees on one."""
        # A "nothing" frame is evidence too, and has to occupy a slot in the
        # window -- otherwise a gesture held for four frames an hour apart
        # would settle, and a lowered hand would never un-settle.
        self._candidates.append(candidate or GestureCandidate("NONE", 0.0))
        if len(self._candidates) > self.window_frames:
            self._candidates = self._candidates[-self.window_frames :]

        settled = vote_gesture(self._candidates, min_frames_agreed=self.min_frames_agreed)
        if settled is None or settled.code == "NONE":
            return None
        return settled

    def reset(self) -> None:
        self._candidates.clear()
