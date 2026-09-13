"""EVQM (§7.2). Blocker #5: hysteresis must prevent profile flapping."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from drishti_worker.evqm import (
    DAY,
    DEGRADED,
    FOG,
    LOWLIGHT,
    NIGHT,
    EVQMConfig,
    ProfileHysteresis,
    profile_vote,
)
from drishti_worker.types import QualityMetrics

CFG = EVQMConfig()


def qm(brightness=0.6, contrast=0.4, blur=0.9, fog=0.1, noise=0.1, motion=0.0):
    return QualityMetrics(brightness, contrast, blur, fog, noise, motion)


class TestVoting:
    def test_day(self):
        assert profile_vote(qm(), CFG) == DAY

    def test_lowlight_then_night(self):
        assert profile_vote(qm(brightness=0.30), CFG) == LOWLIGHT
        assert profile_vote(qm(brightness=0.10), CFG) == NIGHT

    def test_fog(self):
        assert profile_vote(qm(fog=0.8), CFG) == FOG

    def test_blur_and_noise_mean_degraded(self):
        assert profile_vote(qm(blur=0.05), CFG) == DEGRADED
        assert profile_vote(qm(noise=0.9), CFG) == DEGRADED

    def test_structural_degradation_outranks_darkness(self):
        """When the image itself is broken no enhancement rescues it, and the
        risk model should be told to discount what it sees."""
        assert profile_vote(qm(brightness=0.05, blur=0.02), CFG) == DEGRADED

    def test_fog_at_night_is_still_fog(self):
        assert profile_vote(qm(brightness=0.15, fog=0.9), CFG) == FOG

    @given(
        b=st.floats(0, 1),
        c=st.floats(0, 1),
        bl=st.floats(0, 1),
        f=st.floats(0, 1),
        n=st.floats(0, 1),
    )
    @settings(max_examples=300)
    def test_always_returns_a_known_profile(self, b, c, bl, f, n):
        assert profile_vote(qm(b, c, bl, f, n), CFG) in (
            DAY,
            LOWLIGHT,
            NIGHT,
            FOG,
            DEGRADED,
        )


class TestHysteresis:
    def test_headlight_sweep_does_not_flip(self):
        """Blocker #5, stated as a test. Alternating votes must never change
        the profile, however long they alternate."""
        h = ProfileHysteresis(enter_samples=3, exit_samples=5)
        for i in range(200):
            h.submit(NIGHT if i % 2 else DAY)
        assert h.current == DAY
        assert h.changes == 0

    def test_sustained_change_is_accepted(self):
        h = ProfileHysteresis(enter_samples=3, exit_samples=5)
        for _ in range(5):
            h.submit(NIGHT)
        assert h.current == NIGHT
        assert h.changes == 1

    def test_entering_needs_both_counters(self):
        """Three votes satisfy enter_samples but not exit_samples; the incumbent
        holds until it has genuinely lost ground."""
        h = ProfileHysteresis(enter_samples=3, exit_samples=5)
        for _ in range(3):
            h.submit(NIGHT)
        assert h.current == DAY

    def test_a_single_reaffirmation_resets_the_challenge(self):
        h = ProfileHysteresis(enter_samples=3, exit_samples=5)
        for _ in range(4):
            h.submit(NIGHT)
        h.submit(DAY)  # incumbent reaffirmed
        for _ in range(4):
            h.submit(NIGHT)
        assert h.current == DAY  # the streak restarted, so it has not flipped yet

    def test_alternating_challengers_never_win(self):
        """Votes that differ each time never accumulate an entry streak."""
        h = ProfileHysteresis(enter_samples=3, exit_samples=5)
        for i in range(60):
            h.submit([NIGHT, FOG, DEGRADED][i % 3])
        assert h.current == DAY

    def test_unknown_vote_raises(self):
        with pytest.raises(ValueError):
            ProfileHysteresis().submit("twilight")

    def test_bad_counts_raise(self):
        with pytest.raises(ValueError):
            ProfileHysteresis(enter_samples=0)

    @given(votes=st.lists(st.sampled_from([DAY, NIGHT, FOG, DEGRADED]), max_size=200))
    @settings(max_examples=200)
    def test_changes_never_exceed_votes_over_exit_samples(self, votes):
        """A bound on flappiness: with exit_samples=5 you cannot change profile
        more than once per 5 votes, whatever the input."""
        h = ProfileHysteresis(enter_samples=3, exit_samples=5)
        for v in votes:
            h.submit(v)
        assert h.changes <= len(votes) // 5 + 1


def test_config_from_mapping():
    cfg = EVQMConfig.from_mapping(
        {
            "evqm": {
                "hysteresis": {"enter_samples": 2, "exit_samples": 9},
                "thresholds": {"fog_above": 0.4},
            }
        }
    )
    assert cfg.enter_samples == 2
    assert cfg.exit_samples == 9
    assert cfg.fog_above == 0.4
    assert cfg.sample_every_n == 15  # default survives
