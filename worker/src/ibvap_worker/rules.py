"""Rule engine and debouncer (BUILD_SPEC §7.7). **[DEMO-CRITICAL]**

Rules turn geometry into ``Signal``s. The debouncer decides which of those
deserve an operator's attention.

Blocker #1, quoting CLAUDE.md:

    Alert spam. No debounce -> a person standing on a tripwire generates dozens
    of alerts per minute. Debouncing, correlation, and min_track_age are
    requirements, not polish.

Three mechanisms do that work, and all three matter:

1. **Gating** (§7.7.1) — nothing fires on a track younger than
   ``min_track_age_frames`` or with fewer than ``min_hits`` matches. Detector
   flicker never reaches the rules at all.
2. **Debouncing** — the same ``(camera, track, zone, rule)`` is suppressed for
   ``cooldown_s``. The key includes the track id, so a *second* intruder is
   never silenced by the first. Getting that key wrong is the classic way to
   turn a spam bug into a missed-intrusion bug.
3. **Correlation** — several rules firing on the same track within
   ``correlate_window_s`` become one alert carrying several reason codes. This
   is also what makes the risk breakdown rich rather than repetitive: intrusion
   *and* night *and* loitering is one alert at high severity, not three.

There is no automated response anywhere in this module (P7). Rules produce
signals; the operator decides.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dtime
from enum import StrEnum
from typing import Any, Protocol

from .geometry import (
    crossing_direction,
    dwell_seconds,
    perpendicular_travel,
    point_in_polygon,
    point_to_segment_distance,
    speed_body_heights_per_s,
    speed_m_per_s,
)
from .group import classify_group_motion, spread_series
from .risk import RiskConfig, loiter_signal
from .types import CameraRuntime, Signal, Track, ZoneKind, ZoneRuntime

logger = logging.getLogger(__name__)

__all__ = [
    "AlertDecision",
    "DebounceConfig",
    "Debouncer",
    "Decision",
    "RuleConfig",
    "RuleContext",
    "RuleEngine",
    "TrackZoneState",
]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class TrackZoneState:
    """Per-track memory. Rules are otherwise stateless.

    ``zones_last_frame`` is what makes an *entry* distinguishable from
    *being inside*: without it, ZONE_INTRUSION fires on every frame a person
    stands in a zone, and the debouncer has to mop up an avoidable mess.
    """

    zones_last_frame: set[str] = field(default_factory=set)
    fired_once: set[str] = field(default_factory=set)  # rule codes that already fired
    approach_distances: dict[str, deque[float]] = field(default_factory=dict)
    last_foot_point: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class RuleContext:
    camera: CameraRuntime
    zones: Sequence[ZoneRuntime]
    now: datetime
    profile: str  # EVQM profile — rules may soften at night
    prev_state: TrackZoneState
    fps: float = 6.0
    all_tracks: Sequence[Track] = ()
    tamper_suspected: bool = False
    plate_hit: Mapping[str, Any] | None = None
    face_hit: Mapping[str, Any] | None = None
    #: Settled hand signal for ONE track (§7.7, gestures). A single slot, like
    #: ``plate_hit`` -- the rule checks the track_id inside it matches.
    gesture_hit: Mapping[str, Any] | None = None
    #: Settled weapon detection for ONE track (§7.7, weapons). Same contract.
    weapon_hit: Mapping[str, Any] | None = None
    in_patrol_window: bool = False


class Rule(Protocol):
    code: str
    #: A contextual rule enriches an alert but never raises one by itself.
    #: NIGHT_MOVEMENT is the motivating case: "a person is visible at night" is
    #: not an event at a post where patrols walk around every hour, but it does
    #: make a tripwire crossing more serious. Letting it fire alone would
    #: produce an alert every 45 seconds all night, which is blocker #1 in a
    #: different costume. P3.
    standalone: bool

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleConfig:
    min_hits: int = 3
    min_track_age_frames: int = 8
    min_track_conf: float = 0.45
    drop_inside_mask_zones: bool = True

    zone_intrusion: bool = True
    zone_intrusion_classes: tuple[str, ...] = ("person", "vehicle")
    tripwire: bool = True
    tripwire_classes: tuple[str, ...] = ("person", "vehicle")
    min_crossing_px: float = 4.0
    loiter: bool = True
    loiter_classes: tuple[str, ...] = ("person",)
    loiter_seconds: float = 30.0
    loiter_step_seconds: float = 30.0
    night_movement: bool = True
    night_profiles: tuple[str, ...] = ("night", "lowlight")
    night_from: str = "18:30"
    night_to: str = "06:00"
    perimeter_approach: bool = True
    approach_px: float = 80.0
    approach_monotonic_frames: int = 5
    unauthorised_vehicle: bool = True
    watchlist_plate: bool = True
    watchlist_face: bool = False
    crowd_forming: bool = True
    crowd_min: int = 4
    group_motion: bool = True
    group_min: int = 3
    group_window_frames: int = 12
    group_converge_ratio: float = 0.70
    group_disperse_ratio: float = 1.40
    group_min_spread_px: float = 60.0
    hand_signal: bool = True
    hand_signal_codes: tuple[str, ...] = ("HANDS_UP", "ARM_RAISED", "POINTING")
    weapon_visible: bool = True
    fast_movement: bool = True
    fast_movement_classes: tuple[str, ...] = ("person",)
    #: Body-heights per second. ~0.8 is walking, ~2.9 is running (geometry.py).
    fast_heights_per_s: float = 2.0
    #: Above this it is not a human, it is a tracker id switch.
    fast_implausible_heights_per_s: float = 8.0
    fast_window_frames: int = 5
    camera_tamper: bool = True

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> RuleConfig:
        block = dict(cfg.get("rules", cfg))
        gating = dict(block.get("gating", {}))

        def sub(name: str) -> dict[str, Any]:
            return dict(block.get(name, {}) or {})

        zi, tw, lo = sub("zone_intrusion"), sub("tripwire_cross"), sub("loiter")
        nm, pa = sub("night_movement"), sub("perimeter_approach")
        uv, wp, wf = (
            sub("unauthorised_vehicle"),
            sub("watchlist_plate"),
            sub("watchlist_face"),
        )
        cf, ct = sub("crowd_forming"), sub("camera_tamper")
        gm, hs, wv = sub("group_motion"), sub("hand_signal"), sub("weapon_visible")
        fm = sub("fast_movement")
        window = dict(nm.get("window", {}))

        return cls(
            min_hits=int(gating.get("min_hits", 3)),
            min_track_age_frames=int(gating.get("min_track_age_frames", 8)),
            min_track_conf=float(gating.get("min_track_conf", 0.45)),
            drop_inside_mask_zones=bool(gating.get("drop_inside_mask_zones", True)),
            zone_intrusion=bool(zi.get("enabled", True)),
            zone_intrusion_classes=tuple(zi.get("classes", ("person", "vehicle"))),
            tripwire=bool(tw.get("enabled", True)),
            tripwire_classes=tuple(tw.get("classes", ("person", "vehicle"))),
            min_crossing_px=float(tw.get("min_crossing_px", 4.0)),
            loiter=bool(lo.get("enabled", True)),
            loiter_classes=tuple(lo.get("classes", ("person",))),
            loiter_seconds=float(lo.get("seconds", 30.0)),
            loiter_step_seconds=float(lo.get("extra_step_seconds", 30.0)),
            night_movement=bool(nm.get("enabled", True)),
            night_profiles=tuple(nm.get("evqm_profiles", ("night", "lowlight"))),
            night_from=str(window.get("from", "18:30")),
            night_to=str(window.get("to", "06:00")),
            perimeter_approach=bool(pa.get("enabled", True)),
            approach_px=float(pa.get("approach_px", 80.0)),
            approach_monotonic_frames=int(pa.get("monotonic_frames", 5)),
            unauthorised_vehicle=bool(uv.get("enabled", True)),
            watchlist_plate=bool(wp.get("enabled", True)),
            watchlist_face=bool(wf.get("enabled", False)),
            crowd_forming=bool(cf.get("enabled", True)),
            crowd_min=int(cf.get("crowd_min", 4)),
            group_motion=bool(gm.get("enabled", True)),
            group_min=int(gm.get("group_min", 3)),
            group_window_frames=int(gm.get("window_frames", 12)),
            group_converge_ratio=float(gm.get("converge_ratio", 0.70)),
            group_disperse_ratio=float(gm.get("disperse_ratio", 1.40)),
            group_min_spread_px=float(gm.get("min_spread_px", 60.0)),
            hand_signal=bool(hs.get("enabled", True)),
            hand_signal_codes=tuple(hs.get("codes", ("HANDS_UP", "ARM_RAISED", "POINTING"))),
            weapon_visible=bool(wv.get("enabled", True)),
            fast_movement=bool(fm.get("enabled", True)),
            fast_movement_classes=tuple(fm.get("classes", ("person",))),
            fast_heights_per_s=float(fm.get("heights_per_s", 2.0)),
            fast_implausible_heights_per_s=float(fm.get("implausible_heights_per_s", 8.0)),
            fast_window_frames=int(fm.get("window_frames", 5)),
            camera_tamper=bool(ct.get("enabled", True)),
        )


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def _parse_hhmm(text: str) -> dtime:
    try:
        hh, mm = text.split(":")
        return dtime(int(hh), int(mm))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"bad time {text!r}, expected HH:MM") from exc


def in_window(now: datetime, start: str, end: str) -> bool:
    """Is ``now`` inside a possibly midnight-spanning window?

    ``18:30`` to ``06:00`` is the common case here and the one a naive
    ``start <= t <= end`` gets backwards.
    """
    t = now.time()
    s, e = _parse_hhmm(start), _parse_hhmm(end)
    if s <= e:
        return s <= t <= e
    return t >= s or t <= e


def zone_active(zone: ZoneRuntime, now: datetime) -> bool:
    """A zone outside its schedule contributes nothing (§7.7.3)."""
    if not zone.enabled:
        return False
    schedule = zone.schedule
    if not schedule:
        return True
    windows = schedule.get("windows") or []
    if not windows:
        return True
    weekday = now.weekday()
    for w in windows:
        days = w.get("days")
        if days is not None and weekday not in days:
            continue
        if in_window(now, w.get("from", "00:00"), w.get("to", "23:59")):
            return True
    return False


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class ZoneIntrusionRule:
    standalone = True
    code = "ZONE_INTRUSION"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.zone_intrusion or track.cls not in self.cfg.zone_intrusion_classes:
            return None
        foot = track.foot_point
        for zone in ctx.zones:
            if zone.kind is not ZoneKind.AREA or not zone_active(zone, ctx.now):
                continue
            if zone.classes and track.cls not in zone.classes:
                continue
            if not point_in_polygon(foot, zone.polygon):
                continue
            # Only on ENTRY. Being inside is not news every frame.
            if zone.zone_id in ctx.prev_state.zones_last_frame:
                continue
            weight = self.risk.weight("ZONE_INTRUSION") * (zone.severity_base / 3.0)
            return Signal(
                self.code,
                round(weight, 2),
                {
                    "zone_id": zone.zone_id,
                    "zone_name": zone.name,
                    "severity_base": zone.severity_base,
                    "foot_point": [round(foot[0], 1), round(foot[1], 1)],
                    "class": track.cls,
                },
            )
        return None


class TripwireRule:
    standalone = True
    code = "TRIPWIRE_CROSS"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.tripwire or track.cls not in self.cfg.tripwire_classes:
            return None
        prev = track.prev_foot_point
        cur = track.foot_point
        if prev is None:
            return None

        for zone in ctx.zones:
            if zone.kind is not ZoneKind.TRIPWIRE or not zone_active(zone, ctx.now):
                continue
            if zone.classes and track.cls not in zone.classes:
                continue
            # Jitter suppression, measured PERPENDICULAR to this wire. Using
            # total motion here is a trap: at 6 fps a walking person covers only
            # ~10 px per frame, so a total-motion threshold above that silently
            # discards every genuine crossing while still admitting a track that
            # shuffles sideways along the line. Keep min_crossing_px well below
            # one frame of walking travel at the configured analytics_fps.
            if perpendicular_travel(prev, cur, zone.wire) < self.cfg.min_crossing_px:
                continue
            direction = crossing_direction(prev, cur, zone.wire)
            if direction is None:
                continue
            wanted = zone.direction or "both"
            if wanted != "both" and wanted != direction:
                continue
            key = "TRIPWIRE_CROSS_IN" if direction == "in" else "TRIPWIRE_CROSS_OUT"
            return Signal(
                self.code,
                round(self.risk.weight(key), 2),
                {
                    "zone_id": zone.zone_id,
                    "zone_name": zone.name,
                    "direction": direction,
                    "from": [round(prev[0], 1), round(prev[1], 1)],
                    "to": [round(cur[0], 1), round(cur[1], 1)],
                },
            )
        return None


class LoiterRule:
    standalone = True
    code = "LOITER"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.loiter or track.cls not in self.cfg.loiter_classes:
            return None
        for zone in ctx.zones:
            if zone.kind is not ZoneKind.AREA or not zone_active(zone, ctx.now):
                continue
            dwell = dwell_seconds(track, zone.polygon, ctx.fps)
            if dwell < self.cfg.loiter_seconds:
                continue
            signal = loiter_signal(dwell, self.risk, self.cfg.loiter_step_seconds)
            return Signal(
                signal.code,
                signal.weight,
                {**signal.detail, "zone_id": zone.zone_id, "zone_name": zone.name},
            )
        return None


class NightMovementRule:
    standalone = False  # context, never a standalone alert
    code = "NIGHT_MOVEMENT"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.night_movement or track.cls != "person":
            return None
        # Both the image AND the clock must agree. EVQM alone would fire inside
        # a dark warehouse at noon; the clock alone would fire under floodlights.
        if ctx.profile not in self.cfg.night_profiles:
            return None
        if not in_window(ctx.now, self.cfg.night_from, self.cfg.night_to):
            return None
        return Signal(
            self.code,
            round(self.risk.weight("NIGHT_MOVEMENT"), 2),
            {
                "evqm_profile": ctx.profile,
                "local_time": ctx.now.strftime("%H:%M"),
                "window": f"{self.cfg.night_from}-{self.cfg.night_to}",
            },
        )


class PerimeterApproachRule:
    standalone = True
    code = "PERIMETER_APPROACH"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.perimeter_approach or track.cls != "person":
            return None
        foot = track.foot_point
        for zone in ctx.zones:
            if zone.kind is not ZoneKind.TRIPWIRE or not zone_active(zone, ctx.now):
                continue
            a, b = zone.wire
            dist = point_to_segment_distance(foot, a, b)
            history = ctx.prev_state.approach_distances.setdefault(
                zone.zone_id, deque(maxlen=self.cfg.approach_monotonic_frames + 1)
            )
            history.append(dist)
            if len(history) <= self.cfg.approach_monotonic_frames:
                continue
            # Closing, consistently, and now near. Someone walking parallel to
            # the fence never satisfies the monotonic condition.
            values = list(history)
            closing = all(values[i] > values[i + 1] for i in range(len(values) - 1))
            if closing and dist < self.cfg.approach_px:
                return Signal(
                    self.code,
                    round(self.risk.weight("PERIMETER_APPROACH"), 2),
                    {
                        "zone_id": zone.zone_id,
                        "zone_name": zone.name,
                        "distance_px": round(dist, 1),
                        "frames_closing": len(values) - 1,
                    },
                )
        return None


class CrowdFormingRule:
    standalone = True
    code = "CROWD_FORMING"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.crowd_forming or track.cls != "person":
            return None
        for zone in ctx.zones:
            if zone.kind is not ZoneKind.AREA or not zone_active(zone, ctx.now):
                continue
            if not point_in_polygon(track.foot_point, zone.polygon):
                continue
            inside = [
                t
                for t in ctx.all_tracks
                if t.cls == "person"
                and t.is_confirmed
                and point_in_polygon(t.foot_point, zone.polygon)
            ]
            if len(inside) < self.cfg.crowd_min:
                continue
            # Report once, on the lowest track id, so a crowd of six produces
            # one signal and not six.
            if track.track_id != min(t.track_id for t in inside):
                return None
            return Signal(
                self.code,
                round(self.risk.weight("CROWD_FORMING"), 2),
                {
                    "zone_id": zone.zone_id,
                    "zone_name": zone.name,
                    "person_count": len(inside),
                    "threshold": self.cfg.crowd_min,
                },
            )
        return None


class GroupMotionRule:
    """Coordinated movement of several people (§7.7, group behaviour).

    Sits next to CrowdFormingRule deliberately and answers the other half of
    the question: that rule counts heads inside a zone, this one watches what
    the heads are doing. Camera-wide rather than zone-bound, because a group
    converging on a fence is outside every zone right up until the moment the
    warning stops being useful.
    """

    standalone = True
    code = "GROUP_MOTION"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.group_motion or track.cls != "person":
            return None

        people = [t for t in ctx.all_tracks if t.cls == "person" and t.is_confirmed]
        if len(people) < self.cfg.group_min:
            return None
        # One signal per group, not one per member -- same reasoning (and the
        # same lowest-id convention) as CrowdFormingRule. Checked before the
        # geometry so five people do not each pay for the same computation.
        if track.track_id != min(t.track_id for t in people):
            return None

        series = spread_series(people, self.cfg.group_window_frames)
        if not series:
            return None
        # spread_series drops members without enough history, so re-check the
        # count against what actually contributed rather than what was offered.
        contributing = sum(1 for t in people if len(t.history) >= self.cfg.group_window_frames)
        if contributing < self.cfg.group_min:
            return None

        motion = classify_group_motion(
            series,
            contributing,
            converge_ratio=self.cfg.group_converge_ratio,
            disperse_ratio=self.cfg.group_disperse_ratio,
            min_spread_px=self.cfg.group_min_spread_px,
        )
        if motion is None:
            return None

        return Signal(
            motion.code,
            round(self.risk.weight(motion.code), 2),
            {
                "person_count": motion.members,
                "spread_from_px": round(float(motion.spread_from), 1),
                "spread_to_px": round(float(motion.spread_to), 1),
                "ratio": round(float(motion.ratio), 3),
                "window_frames": self.cfg.group_window_frames,
                "window_seconds": round(self.cfg.group_window_frames / max(ctx.fps, 1e-6), 1),
            },
        )


class HandSignalRule:
    """A settled hand signal on a tracked person (§7.7, gestures).

    CONTEXTUAL, never standalone, and the reasoning is NIGHT_MOVEMENT's almost
    word for word (P3): a person waving in the middle of a field is not an
    event. Letting this raise on its own would fire on every soldier beckoning
    a colleague and every civilian shading their eyes — blocker #1 in yet
    another costume. What it *is* good for is making a real event legible:
    "crossed the tripwire" and "crossed the tripwire while signalling to
    somebody off-camera" are different situations, and the second one belongs
    in the operator's breakdown.

    HANDS_UP carries a NEGATIVE weight and that is deliberate, not a bug. A
    person showing empty hands is de-escalating, and an additive model that
    cannot say so is one that only ever shouts louder (P3, and the same
    mechanism as KNOWN_PATROL_WINDOW).
    """

    standalone = False
    code = "HAND_SIGNAL"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.hand_signal or ctx.gesture_hit is None:
            return None
        hit = ctx.gesture_hit
        if hit.get("track_id") != track.track_id:
            return None
        code = str(hit.get("code", ""))
        if code not in self.cfg.hand_signal_codes:
            return None
        return Signal(
            code,
            round(self.risk.weight(code), 2),
            {
                "gesture": code,
                "gesture_conf": hit.get("conf"),
                "frames_agreed": hit.get("frames_agreed"),
            },
        )


class FastMovementRule:
    """Someone moving much faster than a walk (§7.7).

    Running at a fence line is worth an operator's attention on its own — a
    rush at a weak point, or somebody breaking away from a patrol — so this is
    standalone. It is cheap in the way that matters: the tracker has already
    measured this motion, so there is no model, no crop and no extra inference.

    Measured in body-heights per second, never pixels (see
    ``speed_body_heights_per_s``). A pixel threshold would really be a
    threshold on how near the camera somebody is.

    **The guard that matters is the upper one.** A tracker id switch between
    two people standing apart produces an apparent jump of hundreds of pixels
    in one frame — arbitrarily fast, and always fast enough to clear any
    "running" threshold. Reporting that as a sprinting intruder would be
    inventing an event out of a bookkeeping error, so anything above a speed no
    human can sustain is discarded as the artefact it is rather than escalated.
    """

    standalone = True
    code = "FAST_MOVEMENT"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.fast_movement or track.cls not in self.cfg.fast_movement_classes:
            return None
        if len(track.history) < 2:
            return None

        heights_per_s = speed_body_heights_per_s(track, ctx.fps, self.cfg.fast_window_frames)
        if heights_per_s < self.cfg.fast_heights_per_s:
            return None
        if heights_per_s >= self.cfg.fast_implausible_heights_per_s:
            # Not a person. A track that teleports is a tracker artefact, and
            # dropping it silently here is right: it is not evidence of
            # anything except that association failed for one frame.
            logger.debug(
                "camera=%s track=%s discarded implausible speed %.1f heights/s "
                "(likely an id switch)",
                ctx.camera.code,
                track.track_id,
                heights_per_s,
            )
            return None

        detail: dict[str, Any] = {
            "heights_per_s": round(float(heights_per_s), 2),
            "threshold": self.cfg.fast_heights_per_s,
            "class": track.cls,
        }
        # A calibrated camera can say this in real units. An uncalibrated one
        # says nothing rather than inventing a scale (P3).
        metres = speed_m_per_s(track, ctx.fps, ctx.camera.calibration, self.cfg.fast_window_frames)
        if metres is not None:
            detail["speed_m_per_s"] = round(float(metres), 2)

        return Signal(self.code, round(self.risk.weight(self.code), 2), detail)


class WeaponVisibleRule:
    """A weapon confirmed on a tracked person (§7.7, weapons).

    STANDALONE, unlike HandSignalRule — and that asymmetry is the point. A
    person waving is not an event; a person carrying a firearm at a border post
    is one whether or not they have crossed anything yet. Waiting for them to
    also trip a wire would be waiting for the thing the alert exists to prevent.

    The false-positive discipline lives upstream, in weapon.py: a stricter
    confidence floor than the detector's own, and multi-frame voting, because
    an umbrella at forty metres is a pistol for exactly one frame. P3's
    negative modifiers (LOW_CONFIDENCE, DEGRADED_INPUT, SHORT_TRACK) then apply
    on top automatically, so a weak sighting in fog on a two-second-old track
    scores visibly lower than a clean one in daylight.
    """

    standalone = True
    code = "WEAPON_VISIBLE"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.weapon_visible or ctx.weapon_hit is None:
            return None
        hit = ctx.weapon_hit
        if hit.get("track_id") != track.track_id:
            return None
        return Signal(
            self.code,
            round(self.risk.weight(self.code), 2),
            {
                "weapon_type": hit.get("weapon_type"),
                "weapon_conf": hit.get("conf"),
                "frames_agreed": hit.get("frames_agreed"),
                "frames_seen": hit.get("frames_seen"),
            },
        )


class UnauthorisedVehicleRule:
    standalone = True
    code = "UNAUTHORISED_VEHICLE"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.unauthorised_vehicle or track.cls != "vehicle":
            return None
        for zone in ctx.zones:
            if zone.kind is not ZoneKind.AREA or not zone_active(zone, ctx.now):
                continue
            if not point_in_polygon(track.foot_point, zone.polygon):
                continue
            if not zone.classes or "vehicle" in zone.classes:
                continue  # vehicles are expected here
            if zone.zone_id in ctx.prev_state.zones_last_frame:
                continue
            return Signal(
                self.code,
                round(self.risk.weight("UNAUTHORISED_VEHICLE"), 2),
                {
                    "zone_id": zone.zone_id,
                    "zone_name": zone.name,
                    "permitted_classes": list(zone.classes),
                    "vehicle_type": track.attributes.get("vehicle_type"),
                },
            )
        return None


class WatchlistPlateRule:
    standalone = True
    code = "WATCHLIST_PLATE"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        if not self.cfg.watchlist_plate or ctx.plate_hit is None:
            return None
        hit = ctx.plate_hit
        if hit.get("track_id") != track.track_id:
            return None
        # Detail carries the HMAC, never the plate text (P6). Plaintext is added
        # later, only into a fired alert's evidence document.
        return Signal(
            self.code,
            round(self.risk.weight("WATCHLIST_PLATE"), 2),
            {
                "plate_hmac": hit.get("plate_hmac"),
                "category": hit.get("category"),
                "frames_agreed": hit.get("frames_agreed"),
            },
        )


class WatchlistFaceRule:
    standalone = True
    code = "WATCHLIST_FACE"

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg, self.risk = cfg, risk

    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None:
        # Follows faces.enabled. Disabled is the default and the answer to the
        # privacy question (P6).
        if not self.cfg.watchlist_face or ctx.face_hit is None:
            return None
        hit = ctx.face_hit
        if hit.get("track_id") != track.track_id:
            return None
        return Signal(
            self.code,
            round(self.risk.weight("WATCHLIST_FACE"), 2),
            {
                "person_ref": hit.get("ref_code"),
                "similarity": hit.get("similarity"),
                "category": hit.get("category"),
            },
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RuleEngine:
    """Evaluates every enabled rule against every gated track."""

    def __init__(self, cfg: RuleConfig, risk: RiskConfig) -> None:
        self.cfg = cfg
        self.risk = risk
        self.rules: list[Rule] = [
            ZoneIntrusionRule(cfg, risk),
            TripwireRule(cfg, risk),
            LoiterRule(cfg, risk),
            NightMovementRule(cfg, risk),
            PerimeterApproachRule(cfg, risk),
            CrowdFormingRule(cfg, risk),
            GroupMotionRule(cfg, risk),
            HandSignalRule(cfg, risk),
            FastMovementRule(cfg, risk),
            WeaponVisibleRule(cfg, risk),
            UnauthorisedVehicleRule(cfg, risk),
            WatchlistPlateRule(cfg, risk),
            WatchlistFaceRule(cfg, risk),
        ]
        self._states: dict[int, TrackZoneState] = defaultdict(TrackZoneState)

    def gate(self, track: Track) -> tuple[bool, str]:
        """§7.7.1. Returns (passes, reason_if_not).

        This single check removes most detector flicker, which is most false
        alerts. P3.
        """
        if track.hits < self.cfg.min_hits:
            return False, f"hits {track.hits} < {self.cfg.min_hits}"
        if track.age_frames < self.cfg.min_track_age_frames:
            return False, f"age {track.age_frames} < {self.cfg.min_track_age_frames}"
        if track.max_conf < self.cfg.min_track_conf:
            return False, f"max_conf {track.max_conf:.2f} < {self.cfg.min_track_conf}"
        return True, ""

    def in_mask(self, track: Track, zones: Sequence[ZoneRuntime]) -> bool:
        """§7.7.2. A detection inside a mask zone does not exist."""
        if not self.cfg.drop_inside_mask_zones:
            return False
        foot = track.foot_point
        return any(
            z.kind is ZoneKind.MASK and z.enabled and point_in_polygon(foot, z.polygon)
            for z in zones
        )

    def evaluate(
        self,
        tracks: Sequence[Track],
        camera: CameraRuntime,
        zones: Sequence[ZoneRuntime],
        now: datetime,
        profile: str,
        *,
        tamper_suspected: bool = False,
        plate_hit: Mapping[str, Any] | None = None,
        face_hit: Mapping[str, Any] | None = None,
        gesture_hit: Mapping[str, Any] | None = None,
        weapon_hit: Mapping[str, Any] | None = None,
        in_patrol_window: bool = False,
    ) -> dict[int, list[Signal]]:
        """Return ``{track_id: [signals]}`` for this frame."""
        out: dict[int, list[Signal]] = {}

        for track in tracks:
            if self.in_mask(track, zones):
                self._states.pop(track.track_id, None)
                continue
            passes, _reason = self.gate(track)
            if not passes:
                continue

            state = self._states[track.track_id]
            ctx = RuleContext(
                camera=camera,
                zones=zones,
                now=now,
                profile=profile,
                prev_state=state,
                fps=camera.analytics_fps,
                all_tracks=tracks,
                tamper_suspected=tamper_suspected,
                plate_hit=plate_hit,
                face_hit=face_hit,
                gesture_hit=gesture_hit,
                weapon_hit=weapon_hit,
                in_patrol_window=in_patrol_window,
            )

            signals: list[Signal] = []
            raised_standalone = False
            for rule in self.rules:
                try:
                    signal = rule.evaluate(track, ctx)
                except ValueError:
                    # Degenerate geometry on one zone must not kill the frame.
                    # No silent failures: log with full context and carry on.
                    logger.exception(
                        "rule %s failed on camera=%s track=%s; skipping this rule",
                        rule.code,
                        camera.code,
                        track.track_id,
                    )
                    continue
                if signal is not None:
                    signals.append(signal)
                    raised_standalone = raised_standalone or rule.standalone

            self._update_state(track, zones, state)
            # Contextual signals alone are not an alert.
            if signals and raised_standalone:
                out[track.track_id] = signals

        self._prune(tracks)
        return out

    def close_track(self, track_id: int) -> None:
        """Drop per-track rule state. Called from the tracker's close hook."""
        self._states.pop(track_id, None)

    # -- internals ---------------------------------------------------------

    def _update_state(
        self, track: Track, zones: Sequence[ZoneRuntime], state: TrackZoneState
    ) -> None:
        foot = track.foot_point
        state.zones_last_frame = {
            z.zone_id
            for z in zones
            if z.kind is ZoneKind.AREA and z.enabled and point_in_polygon(foot, z.polygon)
        }
        state.last_foot_point = foot

    def _prune(self, tracks: Sequence[Track]) -> None:
        live = {t.track_id for t in tracks}
        for tid in [t for t in self._states if t not in live]:
            del self._states[tid]


# ---------------------------------------------------------------------------
# Debounce — blocker #1
# ---------------------------------------------------------------------------


class Decision(StrEnum):
    EMIT = "emit"
    SUPPRESS = "suppress"
    MERGE = "merge"
    ESCALATE = "escalate"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True, slots=True)
class AlertDecision:
    decision: Decision
    reason: str
    merge_into: str | None = None
    suppressed_count: int = 0

    @property
    def should_write(self) -> bool:
        return self.decision in (Decision.EMIT, Decision.ESCALATE)


@dataclass(frozen=True, slots=True)
class DebounceConfig:
    cooldown_s: float = 45.0
    escalate_after_s: float = 120.0
    correlate_window_s: float = 8.0
    max_alerts_per_camera_per_min: int = 6
    #: How much stronger a signal must be to break out of correlation and raise
    #: its own alert instead of being merged into a weaker one.
    escalation_margin: float = 10.0
    #: Candidates whose primary signal weighs at least this bypass the
    #: per-minute ceiling (cooldown still applies). The ceiling guards the
    #: operator against spam; it must never be what swallows the one alert
    #: that matters. Seen live: a camera whose zones did not fit its scene
    #: spent its 20/min on tripwire noise, and a WEAPON_VISIBLE (75) arriving
    #: then would have been dropped with the rest. 60 = the "high" band.
    rate_limit_exempt_min_weight: float = 60.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> DebounceConfig:
        block = dict(cfg.get("rules", cfg)).get("debounce", {}) or {}
        return cls(
            cooldown_s=float(block.get("cooldown_s", 45.0)),
            escalate_after_s=float(block.get("escalate_after_s", 120.0)),
            correlate_window_s=float(block.get("correlate_window_s", 8.0)),
            max_alerts_per_camera_per_min=int(block.get("max_alerts_per_camera_per_min", 6)),
            escalation_margin=float(block.get("escalation_margin", 10.0)),
            rate_limit_exempt_min_weight=float(block.get("rate_limit_exempt_min_weight", 60.0)),
        )


@dataclass
class _DebounceEntry:
    first_seen: datetime
    last_emitted: datetime
    suppressed: int = 0
    escalations: int = 0


class Debouncer:
    """Suppression, correlation, escalation and a hard rate ceiling.

    The rate ceiling is last-resort, not primary: if it is engaging regularly
    something upstream is wrong, so hitting it logs at WARNING rather than
    silently discarding.
    """

    def __init__(self, cfg: DebounceConfig | None = None) -> None:
        self.cfg = cfg or DebounceConfig()
        self._entries: dict[tuple[str, int, str | None, str], _DebounceEntry] = {}
        self._recent_by_camera: dict[str, deque[datetime]] = defaultdict(deque)
        # (alert_id, when, weight_of_that_alert's_primary_signal)
        self._recent_alert_by_track: dict[tuple[str, int], tuple[str, datetime, float]] = {}

    def submit(
        self,
        *,
        camera_id: str,
        track_id: int,
        zone_id: str | None,
        rule_code: str,
        now: datetime,
        alert_id: str | None = None,
        weight: float = 0.0,
    ) -> AlertDecision:
        """Decide what to do with one candidate.

        ``weight`` is the risk weight of this candidate's primary signal. It is
        what stops correlation from burying an escalation (see below).
        """
        key = (camera_id, track_id, zone_id, rule_code)
        entry = self._entries.get(key)

        # 1. Correlation: another rule already raised an alert for this same
        #    track moments ago. One event, several reason codes (§7.7.4).
        corr_key = (camera_id, track_id)
        recent = self._recent_alert_by_track.get(corr_key)
        if recent is not None:
            prior_id, prior_ts, prior_weight = recent
            within_window = (now - prior_ts).total_seconds() <= self.cfg.correlate_window_s
            # Correlation must never bury an ESCALATION. A person approaching a
            # fence (+18) and then crossing it (+45) is one situation, but the
            # crossing is the headline: merging it into the approach would show
            # the operator "PERIMETER_APPROACH, low" for an actual intrusion.
            # Severity is the thing the operator triages on, so a materially
            # stronger signal always gets to raise. P2/P3.
            escalation = weight > prior_weight + self.cfg.escalation_margin
            if within_window and entry is None and not escalation:
                return AlertDecision(
                    Decision.MERGE,
                    f"correlated into {prior_id} within {self.cfg.correlate_window_s}s",
                    merge_into=prior_id,
                )
            if within_window and escalation:
                logger.info(
                    "camera=%s track=%d %s (%.0f) escalates over the correlated "
                    "alert %s (%.0f); raising rather than merging",
                    camera_id,
                    track_id,
                    rule_code,
                    weight,
                    prior_id,
                    prior_weight,
                )

        # 2. Cooldown / escalation for a repeat of the SAME key.
        if entry is not None:
            since_emit = (now - entry.last_emitted).total_seconds()
            since_first = (now - entry.first_seen).total_seconds()
            if since_emit < self.cfg.cooldown_s:
                entry.suppressed += 1
                return AlertDecision(
                    Decision.SUPPRESS,
                    f"within {self.cfg.cooldown_s}s cooldown " f"({since_emit:.1f}s since last)",
                    suppressed_count=entry.suppressed,
                )
            if since_first >= self.cfg.escalate_after_s:
                # Still happening two minutes later. That is not spam, that is
                # a situation, and it deserves to be raised again, louder.
                if not self._allow_rate(camera_id, now, weight):
                    return self._rate_limited(camera_id)
                entry.last_emitted = now
                entry.escalations += 1
                suppressed = entry.suppressed
                entry.suppressed = 0
                self._note(camera_id, track_id, alert_id, now, weight)
                return AlertDecision(
                    Decision.ESCALATE,
                    f"ongoing for {since_first:.0f}s (escalation #{entry.escalations})",
                    suppressed_count=suppressed,
                )

        # 3. Fresh, or cooldown expired.
        if not self._allow_rate(camera_id, now, weight):
            return self._rate_limited(camera_id)

        if entry is None:
            self._entries[key] = _DebounceEntry(first_seen=now, last_emitted=now)
        else:
            entry.last_emitted = now
            entry.suppressed = 0
        self._note(camera_id, track_id, alert_id, now, weight)
        return AlertDecision(Decision.EMIT, "new event")

    def close_track(self, camera_id: str, track_id: int) -> None:
        """Forget a dead track so ids recycled later start clean."""
        for key in [k for k in self._entries if k[0] == camera_id and k[1] == track_id]:
            del self._entries[key]
        self._recent_alert_by_track.pop((camera_id, track_id), None)

    def stats(self) -> dict[str, int]:
        return {
            "tracked_keys": len(self._entries),
            "suppressed_total": sum(e.suppressed for e in self._entries.values()),
            "escalations_total": sum(e.escalations for e in self._entries.values()),
        }

    # -- internals ---------------------------------------------------------

    def _allow_rate(self, camera_id: str, now: datetime, weight: float = 0.0) -> bool:
        window = self._recent_by_camera[camera_id]
        cutoff = now - timedelta(seconds=60)
        while window and window[0] < cutoff:
            window.popleft()
        if weight >= self.cfg.rate_limit_exempt_min_weight:
            return True
        return len(window) < self.cfg.max_alerts_per_camera_per_min

    def _rate_limited(self, camera_id: str) -> AlertDecision:
        logger.warning(
            "camera=%s hit the alert rate ceiling (%d/min); dropping candidates. "
            "This is a last-resort guard - investigate the upstream cause.",
            camera_id,
            self.cfg.max_alerts_per_camera_per_min,
        )
        return AlertDecision(
            Decision.RATE_LIMITED,
            f"camera exceeded {self.cfg.max_alerts_per_camera_per_min} alerts/min",
        )

    def _note(
        self,
        camera_id: str,
        track_id: int,
        alert_id: str | None,
        now: datetime,
        weight: float = 0.0,
    ) -> None:
        self._recent_by_camera[camera_id].append(now)
        if alert_id:
            self._recent_alert_by_track[(camera_id, track_id)] = (alert_id, now, weight)
