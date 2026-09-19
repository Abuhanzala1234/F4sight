"""Additive risk scoring (BUILD_SPEC §7.8).

P2: *an alert that cannot be explained must not be raised.* The model is additive
on purpose — not because addition is the most accurate way to combine evidence,
but because it is the only way to hand an operator a number and a list of
reasons where the list actually accounts for the number.

The invariant, stated once and enforced in three places (here, the property
tests, and a database constraint):

    round(sum(s.weight for s in breakdown), 2) == score

Clamping would break that, so a clamp emits its own ``CLAMP`` signal carrying the
difference. After clamping, the sum still holds.

Negative weights are not a bug. ``LOW_CONFIDENCE``, ``DEGRADED_INPUT``,
``SHORT_TRACK`` and ``KNOWN_PATROL_WINDOW`` are how the system says *"I saw
something, but look at the conditions"* instead of shouting at full volume. That
is P3 made mechanical.

Pure module: no I/O, no numpy, no clock. Everything comes in through arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import RiskResult, Severity, Signal

__all__ = ["DEFAULT_WEIGHTS", "RiskConfig", "RiskContext", "score", "severity_for"]

# Defaults mirror config/risk.yaml. The config file wins at runtime; these exist
# so the module is importable and testable on its own.
DEFAULT_WEIGHTS: dict[str, float] = {
    "ZONE_INTRUSION": 40.0,
    "TRIPWIRE_CROSS_IN": 45.0,
    "TRIPWIRE_CROSS_OUT": 20.0,
    "LOITER_BASE": 15.0,
    "LOITER_PER_STEP": 5.0,
    "LOITER_CAP": 30.0,
    "NIGHT_MOVEMENT": 20.0,
    "PERIMETER_APPROACH": 18.0,
    "WATCHLIST_FACE": 35.0,
    "WATCHLIST_PLATE": 30.0,
    "CROWD_FORMING": 25.0,
    "GROUP_CONVERGING": 30.0,
    "GROUP_DISPERSING": 22.0,
    "UNAUTHORISED_VEHICLE": 25.0,
    "CAMERA_TAMPER": 30.0,
    "WEAPON_VISIBLE": 75.0,
    "FAST_MOVEMENT": 28.0,
    "ARM_RAISED": 15.0,
    "POINTING": 10.0,
    "HANDS_UP": -12.0,  # de-escalating, see rules.HandSignalRule
    "LOW_CONFIDENCE": -10.0,
    "DEGRADED_INPUT": -8.0,
    "SHORT_TRACK": -12.0,
    "KNOWN_PATROL_WINDOW": -15.0,
}

DEFAULT_BANDS: dict[str, float] = {
    "info": 0.0,
    "low": 20.0,
    "medium": 40.0,
    "high": 60.0,
    "critical": 80.0,
}

CLAMP_CODE = "CLAMP"


@dataclass(frozen=True, slots=True)
class RiskConfig:
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    bands: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    clamp_min: float = 0.0
    clamp_max: float = 100.0
    emit_clamp_signal: bool = True
    low_confidence_below: float = 0.60
    short_track_frames: int = 15
    degraded_profiles: tuple[str, ...] = ("degraded", "fog")

    def weight(self, code: str, default: float = 0.0) -> float:
        return float(self.weights.get(code, default))

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> RiskConfig:
        """Build from the ``risk:`` block of the merged config."""
        risk = dict(cfg.get("risk", cfg))
        mods = dict(risk.get("modifiers", {}))
        clamp = list(risk.get("clamp", [0.0, 100.0]))
        return cls(
            weights={**DEFAULT_WEIGHTS, **dict(risk.get("weights", {}))},
            bands={**DEFAULT_BANDS, **dict(risk.get("severity_bands", {}))},
            clamp_min=float(clamp[0]),
            clamp_max=float(clamp[1]),
            emit_clamp_signal=bool(risk.get("emit_clamp_signal", True)),
            low_confidence_below=float(mods.get("low_confidence_below", 0.60)),
            short_track_frames=int(mods.get("short_track_frames", 15)),
            degraded_profiles=tuple(mods.get("degraded_profiles", ("degraded", "fog"))),
        )


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Everything the scorer is allowed to know. No globals, no clock."""

    config: RiskConfig = field(default_factory=RiskConfig)
    evqm_profile: str = "day"
    track_max_conf: float = 1.0
    track_age_frames: int = 999
    in_patrol_window: bool = False


def severity_for(score_value: float, bands: Mapping[str, float] | None = None) -> str:
    """Map a score to a severity band (§7.8 rule 4).

    Bands are inclusive-lower: >=80 critical, >=60 high, >=40 medium, >=20 low,
    else info.
    """
    b = dict(bands or DEFAULT_BANDS)
    ordered = sorted(b.items(), key=lambda kv: kv[1], reverse=True)
    for name, threshold in ordered:
        if score_value >= threshold:
            return name
    return Severity.INFO.value


def _modifier_signals(ctx: RiskContext) -> list[Signal]:
    """The negative contributions. P3's mechanism (§7.8)."""
    cfg = ctx.config
    out: list[Signal] = []

    if ctx.track_max_conf < cfg.low_confidence_below:
        out.append(
            Signal(
                "LOW_CONFIDENCE",
                cfg.weight("LOW_CONFIDENCE"),
                {
                    "track_max_conf": round(ctx.track_max_conf, 3),
                    "threshold": cfg.low_confidence_below,
                    "note": "detector was not sure about this object",
                },
            )
        )

    if ctx.evqm_profile in cfg.degraded_profiles:
        out.append(
            Signal(
                "DEGRADED_INPUT",
                cfg.weight("DEGRADED_INPUT"),
                {
                    "evqm_profile": ctx.evqm_profile,
                    "note": "image conditions reduce confidence in this detection",
                },
            )
        )

    if ctx.track_age_frames < cfg.short_track_frames:
        out.append(
            Signal(
                "SHORT_TRACK",
                cfg.weight("SHORT_TRACK"),
                {
                    "age_frames": ctx.track_age_frames,
                    "threshold": cfg.short_track_frames,
                    "note": "object was visible too briefly to be sure",
                },
            )
        )

    if ctx.in_patrol_window:
        out.append(
            Signal(
                "KNOWN_PATROL_WINDOW",
                cfg.weight("KNOWN_PATROL_WINDOW"),
                {"note": "movement coincides with a scheduled friendly patrol"},
            )
        )

    return out


def score(signals: Sequence[Signal], ctx: RiskContext | None = None) -> RiskResult:
    """Combine signals into an explainable score.

    Guarantees (all property-tested in ``tests/test_risk.py``):

    1. ``round(sum(weights), 2) == score`` — including after clamping, via the
       synthetic CLAMP signal.
    2. ``clamp_min <= score <= clamp_max``.
    3. Adding a signal of non-negative weight never lowers the score
       (monotonicity), assuming no clamp is active.
    4. ``score(())`` is exactly ``RiskResult(0.0, (), 'info')`` — the empty case
       is not special-cased anywhere else in the system.
    """
    context = ctx or RiskContext()
    cfg = context.config

    if not signals:
        # Rule 4. An empty signal list is a real, valid, zero-risk answer.
        return RiskResult(score=0.0, breakdown=(), severity=Severity.INFO.value)

    combined: list[Signal] = list(signals)
    combined.extend(_modifier_signals(context))

    raw = round(sum(s.weight for s in combined), 2)
    clamped = round(min(max(raw, cfg.clamp_min), cfg.clamp_max), 2)

    if clamped != raw and cfg.emit_clamp_signal:
        # Keep the invariant true after clamping: the clamp is itself a
        # contribution, and we say so rather than quietly losing points.
        delta = round(clamped - raw, 2)
        combined.append(
            Signal(
                CLAMP_CODE,
                delta,
                {
                    "raw_score": raw,
                    "clamped_to": clamped,
                    "bounds": [cfg.clamp_min, cfg.clamp_max],
                    "note": "score clamped to the configured range",
                },
            )
        )

    final = clamped
    result = RiskResult(
        score=final,
        breakdown=tuple(combined),
        severity=severity_for(final, cfg.bands),
    )

    # Belt and braces. This assertion has caught two real bugs; if it ever
    # fires in production we want a loud crash, not a lying dashboard.
    if not result.sums_correctly():
        raise AssertionError(
            f"risk breakdown does not sum to score: "
            f"sum={sum(s.weight for s in result.breakdown)} score={result.score} "
            f"codes={[s.code for s in result.breakdown]}"
        )
    return result


def loiter_signal(dwell_s: float, cfg: RiskConfig, step_s: float = 30.0) -> Signal:
    """LOITER grows with dwell time but is capped (§7.8).

    Someone standing still for an hour is not twenty times more dangerous than
    someone standing still for three minutes, and an uncapped ramp would drown
    every other signal in the breakdown.
    """
    if step_s <= 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    base = cfg.weight("LOITER_BASE")
    per_step = cfg.weight("LOITER_PER_STEP")
    cap = cfg.weight("LOITER_CAP")
    extra_steps = max(0, int(dwell_s // step_s) - 1)
    weight = min(base + per_step * extra_steps, cap)
    return Signal(
        "LOITER",
        round(weight, 2),
        {
            "dwell_seconds": round(dwell_s, 1),
            "extra_steps": extra_steps,
            "capped_at": cap,
        },
    )
