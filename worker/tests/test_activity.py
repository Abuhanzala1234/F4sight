"""Cheap activity gating (§7.7).

The decision logic is a pure function of numbers, so most of this needs no
arrays and no model. The tests that matter are the two guards: a gesture or
weapon already confirmed must not be undone by the subject going still, and a
gate that never fires must still be overridden by the heartbeat.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibvap_worker.activity import (
    ActivityConfig,
    ActivityGate,
    should_run,
    thumbnail_difference,
)


class TestShouldRun:
    def _decide(self, **kw):
        base = dict(
            first_sighting=False,
            diff=0.0,
            frames_since_run=0,
            runs_so_far=99,  # warmed up, unless a test says otherwise
            diff_threshold=0.02,
            max_stale_frames=60,
            min_runs_before_skip=8,
        )
        base.update(kw)
        return should_run(**base)

    def test_a_new_track_is_always_checked(self):
        """Somebody who walks in already holding a weapon and then stands
        perfectly still would otherwise be gated out on their own stillness."""
        run, reason = self._decide(first_sighting=True, diff=0.0)
        assert run and "first sighting" in reason

    def test_motion_runs_the_model(self):
        run, reason = self._decide(diff=0.9)
        assert run and "motion" in reason

    def test_stillness_skips(self):
        run, reason = self._decide(diff=0.001)
        assert not run and "still" in reason

    def test_heartbeat_overrides_stillness(self):
        """A mis-set threshold, compression noise, or someone walking straight
        at the camera must not hold a track in 'nothing changed' forever."""
        run, reason = self._decide(diff=0.0, frames_since_run=60)
        assert run and "heartbeat" in reason

    def test_warmup_forces_runs_before_any_skipping(self):
        """Guard 3, the bug this caught: gestures and weapons need several
        agreeing looks to confirm. A gate that skips immediately starves that
        vote, and a motionless armed person never alerts at all."""
        run, reason = self._decide(diff=0.0, runs_so_far=2, min_runs_before_skip=8)
        assert run and "warming up" in reason

    def test_skipping_resumes_once_warmed_up(self):
        assert not self._decide(diff=0.0, runs_so_far=8, min_runs_before_skip=8)[0]

    def test_threshold_boundary_runs(self):
        assert self._decide(diff=0.02, diff_threshold=0.02)[0]
        assert not self._decide(diff=0.0199, diff_threshold=0.02)[0]


class TestThumbnailDifference:
    def test_identical_is_zero(self):
        a = np.full((32, 32), 128, dtype=np.uint8)
        assert thumbnail_difference(a, a.copy()) == 0.0

    def test_opposite_is_one(self):
        a = np.zeros((32, 32), dtype=np.uint8)
        b = np.full((32, 32), 255, dtype=np.uint8)
        assert thumbnail_difference(a, b) == pytest.approx(1.0)

    def test_is_normalised_and_symmetric(self):
        rng = np.random.default_rng(7)
        a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        d = thumbnail_difference(a, b)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(thumbnail_difference(b, a))

    def test_does_not_overflow_uint8(self):
        """A naive uint8 subtraction wraps around: 0 - 255 becomes 1, and a
        maximally different pair would read as almost identical."""
        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.full((8, 8), 255, dtype=np.uint8)
        assert thumbnail_difference(a, b) > 0.9

    def test_shape_mismatch_is_a_bug_not_a_condition(self):
        with pytest.raises(ValueError):
            thumbnail_difference(np.zeros((8, 8)), np.zeros((16, 16)))


class TestActivityGate:
    def _crop(self, value: int, size: int = 64) -> np.ndarray:
        return np.full((size, size, 3), value, dtype=np.uint8)

    def test_first_call_runs_then_identical_frames_skip(self):
        gate = ActivityGate(ActivityConfig(max_stale_frames=1000, min_runs_before_skip=1))
        assert gate.check(1, self._crop(100), frame_id=0)[0]
        for f in range(1, 6):
            assert not gate.check(1, self._crop(100), frame_id=f)[0]
        assert gate.runs == 1 and gate.skips == 5

    def test_a_changed_crop_runs_again(self):
        gate = ActivityGate(ActivityConfig(max_stale_frames=1000, min_runs_before_skip=1))
        gate.check(1, self._crop(100), frame_id=0)
        assert not gate.check(1, self._crop(100), frame_id=1)[0]
        assert gate.check(1, self._crop(200), frame_id=2)[0]

    def test_heartbeat_fires_on_a_totally_static_track(self):
        gate = ActivityGate(ActivityConfig(max_stale_frames=5, min_runs_before_skip=1))
        gate.check(1, self._crop(100), frame_id=0)
        ran = [gate.check(1, self._crop(100), frame_id=f)[0] for f in range(1, 12)]
        assert any(ran), "heartbeat must override an indefinitely still crop"

    def test_slow_drift_is_measured_against_the_last_look_not_the_last_run(self):
        """One-step-at-a-time drift must not accumulate unseen. Comparing only
        against the last frame the model RAN on would let a subject creep
        across the frame in sub-threshold increments forever."""
        gate = ActivityGate(
            ActivityConfig(diff_threshold=0.05, max_stale_frames=1000, min_runs_before_skip=1)
        )
        gate.check(1, self._crop(100), frame_id=0)
        # Each step is tiny (4/255 ~= 0.016, under the 0.05 threshold).
        ran = [gate.check(1, self._crop(100 + 4 * i), frame_id=i)[0] for i in range(1, 8)]
        assert not any(ran)

    def test_tracks_are_independent(self):
        gate = ActivityGate(ActivityConfig(max_stale_frames=1000, min_runs_before_skip=1))
        gate.check(1, self._crop(100), frame_id=0)
        # A different track's first sighting must not inherit track 1's state.
        assert gate.check(2, self._crop(100), frame_id=0)[0]

    def test_close_track_forgets_it(self):
        gate = ActivityGate(ActivityConfig(max_stale_frames=1000, min_runs_before_skip=1))
        gate.check(1, self._crop(100), frame_id=0)
        gate.close_track(1)
        # Seen as new again, so it is checked rather than skipped.
        assert gate.check(1, self._crop(100), frame_id=1)[0]

    def test_disabled_gate_always_runs(self):
        gate = ActivityGate(ActivityConfig(enabled=False))
        for f in range(5):
            assert gate.check(1, self._crop(100), frame_id=f)[0]
        assert gate.skips == 0

    def test_stats_report_the_saving(self):
        gate = ActivityGate(ActivityConfig(max_stale_frames=1000, min_runs_before_skip=1))
        gate.check(1, self._crop(100), frame_id=0)
        for f in range(1, 4):
            gate.check(1, self._crop(100), frame_id=f)
        assert gate.stats() == {"runs": 1, "skips": 3, "skipped_pct": 75.0}


class TestConfig:
    def test_from_mapping(self):
        cfg = ActivityConfig.from_mapping(
            {"activity_gate": {"enabled": False, "diff_threshold": 0.1, "max_stale_frames": 5}}
        )
        assert not cfg.enabled
        assert cfg.diff_threshold == 0.1
        assert cfg.max_stale_frames == 5

    def test_defaults_are_enabled(self):
        """The gate should be on by default -- it is a pure saving with the
        heartbeat as its safety net."""
        assert ActivityConfig.from_mapping({}).enabled
