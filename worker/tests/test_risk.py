"""Risk scoring (§7.8). The sum invariant is the whole contract."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from drishti_worker.risk import (
    DEFAULT_WEIGHTS,
    RiskConfig,
    RiskContext,
    loiter_signal,
    score,
    severity_for,
)
from drishti_worker.types import RiskResult, Signal

weights = st.floats(min_value=-60, max_value=60, allow_nan=False, allow_infinity=False, width=32)
codes = st.sampled_from(sorted(DEFAULT_WEIGHTS))
signals = st.builds(Signal, code=codes, weight=weights, detail=st.just({}))

# A context whose modifiers are all off, so tests can reason about the signals
# they actually passed in.
NEUTRAL = RiskContext(track_max_conf=1.0, track_age_frames=999, evqm_profile="day")


class TestInvariant:
    @given(sigs=st.lists(signals, min_size=1, max_size=12))
    @settings(max_examples=400)
    def test_breakdown_always_sums_to_score(self, sigs):
        """P2. If these don't add up, the explanation shown to an operator is
        a lie — so this property holds for every input, clamped or not."""
        result = score(sigs, NEUTRAL)
        assert result.sums_correctly(), (
            result.score,
            [(s.code, s.weight) for s in result.breakdown],
        )

    @given(sigs=st.lists(signals, min_size=1, max_size=12))
    @settings(max_examples=400)
    def test_score_is_bounded(self, sigs):
        result = score(sigs, NEUTRAL)
        assert 0.0 <= result.score <= 100.0

    @given(
        sigs=st.lists(signals, min_size=0, max_size=6),
        extra=st.floats(min_value=0, max_value=40, allow_nan=False, width=32),
    )
    @settings(max_examples=300)
    def test_monotonic_in_non_negative_signals(self, sigs, extra):
        """Adding a non-negative signal never lowers the score."""
        before = score([*sigs, Signal("ZONE_INTRUSION", 0.0, {})], NEUTRAL)
        after = score([*sigs, Signal("ZONE_INTRUSION", extra, {})], NEUTRAL)
        if before.score < 100.0:
            assert after.score >= before.score - 1e-6

    def test_empty_is_exactly_zero(self):
        """Rule 4: the empty case is a real answer, not a special case."""
        assert score(()) == RiskResult(0.0, (), "info")

    def test_clamp_emits_its_own_signal(self):
        result = score([Signal("A", 80.0, {}), Signal("B", 70.0, {})], NEUTRAL)
        assert result.score == 100.0
        assert any(s.code == "CLAMP" for s in result.breakdown)
        assert result.sums_correctly()

    def test_negative_clamp_emits_signal(self):
        result = score([Signal("A", -50.0, {})], NEUTRAL)
        assert result.score == 0.0
        assert result.sums_correctly()


class TestModifiers:
    def test_low_confidence_reduces_score(self):
        base = score([Signal("ZONE_INTRUSION", 40.0, {})], NEUTRAL)
        poor = score(
            [Signal("ZONE_INTRUSION", 40.0, {})],
            RiskContext(track_max_conf=0.3, track_age_frames=999),
        )
        assert poor.score < base.score
        assert any(s.code == "LOW_CONFIDENCE" for s in poor.breakdown)

    def test_degraded_input_reduces_score(self):
        result = score(
            [Signal("ZONE_INTRUSION", 40.0, {})],
            RiskContext(evqm_profile="fog", track_age_frames=999),
        )
        assert any(s.code == "DEGRADED_INPUT" for s in result.breakdown)

    def test_short_track_reduces_score(self):
        result = score([Signal("ZONE_INTRUSION", 40.0, {})], RiskContext(track_age_frames=4))
        assert any(s.code == "SHORT_TRACK" for s in result.breakdown)

    def test_patrol_window_reduces_score(self):
        result = score(
            [Signal("TRIPWIRE_CROSS", 45.0, {})],
            RiskContext(in_patrol_window=True, track_age_frames=999),
        )
        assert any(s.code == "KNOWN_PATROL_WINDOW" for s in result.breakdown)

    def test_reason_codes_exclude_negatives(self):
        """Reason codes are what the alert IS, not what discounted it."""
        result = score(
            [Signal("ZONE_INTRUSION", 40.0, {})],
            RiskContext(track_max_conf=0.2, track_age_frames=999),
        )
        assert "ZONE_INTRUSION" in result.reason_codes()
        assert "LOW_CONFIDENCE" not in result.reason_codes()


class TestSeverity:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "info"),
            (19.9, "info"),
            (20, "low"),
            (39.9, "low"),
            (40, "medium"),
            (59.9, "medium"),
            (60, "high"),
            (79.9, "high"),
            (80, "critical"),
            (100, "critical"),
        ],
    )
    def test_bands(self, value, expected):
        assert severity_for(value) == expected

    @given(value=st.floats(0, 100, allow_nan=False))
    @settings(max_examples=200)
    def test_severity_is_monotonic(self, value):
        order = ["info", "low", "medium", "high", "critical"]
        lower = severity_for(max(0.0, value - 25))
        upper = severity_for(value)
        assert order.index(upper) >= order.index(lower)


class TestLoiter:
    def test_capped(self):
        cfg = RiskConfig()
        assert loiter_signal(30.0, cfg).weight == 15.0
        assert loiter_signal(1e6, cfg).weight == cfg.weight("LOITER_CAP")

    @given(dwell=st.floats(0, 5000, allow_nan=False))
    @settings(max_examples=200)
    def test_never_exceeds_cap(self, dwell):
        cfg = RiskConfig()
        assert loiter_signal(dwell, cfg).weight <= cfg.weight("LOITER_CAP")

    def test_bad_step_raises(self):
        with pytest.raises(ValueError):
            loiter_signal(60.0, RiskConfig(), step_s=0)


def test_config_from_mapping():
    cfg = RiskConfig.from_mapping(
        {
            "risk": {
                "weights": {"ZONE_INTRUSION": 99.0},
                "modifiers": {"short_track_frames": 3},
            }
        }
    )
    assert cfg.weight("ZONE_INTRUSION") == 99.0
    assert cfg.short_track_frames == 3
    assert cfg.weight("NIGHT_MOVEMENT") == 20.0  # defaults survive a partial override
