"""Shared value types — the vocabulary of the whole worker (BUILD_SPEC §7).

Everything here is a frozen dataclass with no behaviour beyond derived
properties. If a type in this module grows a method that does I/O, it has
stopped being a value and belongs somewhere else.

This module imports numpy only for type checking, so the pure-function modules
(geometry, risk, evidence, merkle, anpr validation) stay importable and testable
without any native dependency installed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    NDArray = np.ndarray
else:  # pragma: no cover
    NDArray = Any

# Pixels in ORIGINAL frame space unless a docstring says otherwise.
Point = tuple[float, float]
BoxXYXY = tuple[float, float, float, float]


class StreamState(StrEnum):
    """Lifecycle of one camera's RTSP connection (§7.1)."""

    CONNECTING = "connecting"
    LIVE = "live"
    STALLED = "stalled"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ZoneKind(StrEnum):
    AREA = "area"
    TRIPWIRE = "tripwire"
    MASK = "mask"  # negative zone: detections inside it do not exist


# --------------------------------------------------------------------------
# Coordinate transforms
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameTransform:
    """Maps model-input space back to original-frame space.

    INVARIANT (CLAUDE.md): detections are always mapped back to original frame
    coordinates. Inference may run on a resized, enhanced or tiled image, so the
    transform travels with the frame and is applied the moment the detector
    returns. No module downstream of ``pipeline`` ever sees model-space boxes.
    """

    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0
    crop_x: float = 0.0  # offset of the tile within the original frame
    crop_y: float = 0.0

    @staticmethod
    def letterbox(src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> FrameTransform:
        """Aspect-preserving fit of ``src`` into ``dst`` with centred padding."""
        sw, sh = src_wh
        dw, dh = dst_wh
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            raise ValueError(f"degenerate letterbox: src={src_wh} dst={dst_wh}")
        scale = min(dw / sw, dh / sh)
        new_w, new_h = sw * scale, sh * scale
        return FrameTransform(
            scale_x=scale,
            scale_y=scale,
            pad_x=(dw - new_w) / 2.0,
            pad_y=(dh - new_h) / 2.0,
        )

    def to_original(self, b: BoxXYXY) -> BoxXYXY:
        """Model-space box -> original-frame box."""
        x1, y1, x2, y2 = b
        return (
            (x1 - self.pad_x) / self.scale_x + self.crop_x,
            (y1 - self.pad_y) / self.scale_y + self.crop_y,
            (x2 - self.pad_x) / self.scale_x + self.crop_x,
            (y2 - self.pad_y) / self.scale_y + self.crop_y,
        )

    def point_to_original(self, p: Point) -> Point:
        x, y = p
        return (
            (x - self.pad_x) / self.scale_x + self.crop_x,
            (y - self.pad_y) / self.scale_y + self.crop_y,
        )

    @property
    def is_identity(self) -> bool:
        return (
            self.scale_x == 1.0
            and self.scale_y == 1.0
            and self.pad_x == 0.0
            and self.pad_y == 0.0
            and self.crop_x == 0.0
            and self.crop_y == 0.0
        )


IDENTITY_TRANSFORM = FrameTransform(1.0, 1.0)


# --------------------------------------------------------------------------
# Frames and detections
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded frame. ``image`` is the ORIGINAL and is never mutated (P4)."""

    camera_id: str
    frame_id: int  # monotonic per camera, never reused
    ts_utc: datetime
    image: NDArray  # BGR uint8
    width: int
    height: int
    seq_gap: int = 0  # frames dropped by the reader before this one

    @property
    def shape_wh(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True, slots=True)
class RawDetection:
    """What a detector backend returns: MODEL-space box, model class id."""

    cls_id: int
    conf: float
    box: BoxXYXY


@dataclass(frozen=True, slots=True)
class Detection:
    """A detection in ORIGINAL frame coordinates, in our taxonomy."""

    cls: str  # person | vehicle | animal | bag
    conf: float
    box: BoxXYXY
    cls_id: int
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def foot_point(self) -> Point:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])


@dataclass(frozen=True, slots=True)
class Track:
    """One tracked object, as seen at a single instant (§7.5).

    Frozen: the tracker rebuilds these each update rather than mutating, so a
    Track handed to the rule engine can never change underneath it.
    """

    track_id: int
    cls: str
    box: BoxXYXY
    conf: float
    max_conf: float
    age_frames: int  # frames since birth
    hits: int  # frames with a matched detection
    time_since_update: int
    first_seen: datetime
    last_seen: datetime
    history: tuple[Point, ...] = ()  # foot-points, oldest -> newest
    attributes: Mapping[str, Any] = field(default_factory=dict)
    min_hits: int = 3

    @property
    def foot_point(self) -> Point:
        """Bottom-centre of the box: where the object meets the ground.

        Used for every geometric test. The centroid is wrong for this — a tall
        person's centroid is a metre off the ground and crosses a tripwire early.
        """
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= self.min_hits

    @property
    def prev_foot_point(self) -> Point | None:
        return self.history[-2] if len(self.history) >= 2 else None

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.last_seen - self.first_seen).total_seconds())

    def displacement_px(self) -> float:
        if len(self.history) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.history[0], self.history[-1]
        return math.hypot(x1 - x0, y1 - y0)


# --------------------------------------------------------------------------
# Signals, risk, alerts
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signal:
    """One contribution to a risk score. ``weight`` may be negative (P3)."""

    code: str
    weight: float
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskResult:
    """INVARIANT (P2): round(sum(s.weight for s in breakdown), 2) == score.

    The clamp is reported as a synthetic CLAMP signal precisely so that this
    still holds after clamping. If the numbers do not add up, the explanation
    shown to the operator is a lie.
    """

    score: float
    breakdown: tuple[Signal, ...]
    severity: str

    def sums_correctly(self, tol: float = 0.005) -> bool:
        return abs(sum(s.weight for s in self.breakdown) - self.score) <= tol

    def reason_codes(self) -> tuple[str, ...]:
        return tuple(s.code for s in self.breakdown if s.weight > 0)


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """EVQM output (§7.2). All fields normalised to 0..1."""

    brightness: float
    contrast: float
    blur: float  # 1.0 = sharp
    fog: float  # 1.0 = heavy haze
    noise: float
    motion: float

    def as_dict(self) -> dict[str, float]:
        return {
            "brightness": round(self.brightness, 4),
            "contrast": round(self.contrast, 4),
            "blur": round(self.blur, 4),
            "fog": round(self.fog, 4),
            "noise": round(self.noise, 4),
            "motion": round(self.motion, 4),
        }


@dataclass(frozen=True, slots=True)
class Calibration:
    """Optional image->world mapping (§6.2).

    Absent calibration means speed is reported in px/s and speed-based rules are
    DISABLED rather than guessed. A wrong speed is a wrong alert.
    """

    homography: tuple[float, ...] | None = None  # 9 floats, row-major
    px_per_m_at_y: tuple[tuple[float, float], ...] = ()  # [(y_px, metres_per_px)]


@dataclass(frozen=True, slots=True)
class ZoneRuntime:
    """A zone, denormalised into pixel space for one camera resolution."""

    zone_id: str
    name: str
    kind: ZoneKind
    polygon: tuple[Point, ...]  # pixels
    direction: str | None = None  # 'in' | 'out' | 'both'
    classes: tuple[str, ...] = ()
    schedule: Mapping[str, Any] | None = None
    severity_base: int = 3
    enabled: bool = True

    @property
    def wire(self) -> tuple[Point, Point]:
        if self.kind is not ZoneKind.TRIPWIRE:
            raise ValueError(f"zone {self.zone_id} is {self.kind}, not a tripwire")
        if len(self.polygon) != 2:
            raise ValueError(f"tripwire {self.zone_id} needs exactly 2 points")
        return (self.polygon[0], self.polygon[1])


@dataclass(frozen=True, slots=True)
class CameraRuntime:
    camera_id: str
    code: str
    site_id: str
    site_code: str
    timezone: str
    width: int
    height: int
    analytics_fps: float
    calibration: Calibration | None = None


@dataclass(frozen=True, slots=True)
class AlertCandidate:
    """A scored, pre-debounce alert (§7.7.4)."""

    camera: CameraRuntime
    track: Track
    zone_id: str | None
    kind: str
    risk: RiskResult
    signals: tuple[Signal, ...]
    ts_utc: datetime
    frame: Frame | None = None

    @property
    def debounce_key(self) -> tuple[str, int, str | None, str]:
        """(camera_id, track_id, zone_id, rule_code).

        NOT rule_code alone — a second, different intruder must never be
        suppressed by the first (§7.7.4).
        """
        return (self.camera.camera_id, self.track.track_id, self.zone_id, self.kind)
