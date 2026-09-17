"""Cheap activity gating for the expensive second-stage models (§7.7).

The problem this solves, in the terms it was raised in: a five-minute clip at
20 fps is thousands of frames, and in almost all of them nothing is happening.
Running pose estimation and weapon detection across all of them spends most of
the GPU budget confirming that a person who has not moved still has not moved.

So before each expensive model call this asks a much cheaper question -- *has
this person's crop visibly changed since the last time we looked?* -- by
comparing 32x32 greyscale thumbnails. That costs microseconds against the
~33 ms a model call costs, so skipping even a third of calls pays for the
check many times over.

**What this deliberately does NOT gate.** Detection, tracking, zones,
tripwires, loiter and fast-movement all keep running on every analysed frame,
untouched. Those are the safety-critical path, and a motion gate in front of
them is precisely how you miss the intruder who moves slowly enough not to
trip it -- which at a border post is the intruder who is trying not to be seen.
This gates only the *enrichment* stages, where a missed frame costs detail
rather than the event itself.

**Two guards keep the gate honest, and both exist because the naive version is
dangerous:**

1. *A settled state is never cleared by stillness.* An armed person who stops
   moving must not stop being armed. The gate only decides whether to spend
   another inference; it never decides that a previous answer has expired. The
   caller keeps the last result (pipeline.py).
2. *A heartbeat overrides the gate.* Past ``max_stale_frames`` the model runs
   regardless of how still the crop looks. Compression noise, a mis-set
   threshold, or a subject moving directly toward the camera (which changes
   the picture far less than moving across it) could otherwise hold a track in
   "nothing changed" indefinitely. This bounds how stale any answer can get,
   in exchange for a little of the saving.

3. *The gate never skips before the model has reached a verdict.* This one was
   found by a test, and it is the sharpest edge here. Gestures and weapons are
   confirmed by VOTING across several frames -- three agreeing looks before a
   weapon counts as real. A gate that starts skipping immediately starves that
   vote: the model runs once on first sighting, the subject stands still, the
   vote never reaches three, and **a motionless armed person never alerts at
   all.** That is far worse than the cost the gate was added to save. So the
   gate is forbidden from skipping until it has let the model look
   ``min_runs_before_skip`` times -- a full voting window -- after which the
   verdict exists and stillness may legitimately preserve it.

The decision itself is a pure function of numbers, so it is testable without
arrays, a model, or a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ActivityConfig", "ActivityGate", "should_run", "thumbnail_difference"]


@dataclass(frozen=True, slots=True)
class ActivityConfig:
    enabled: bool = True
    #: Thumbnails are square and tiny on purpose: at 32x32 the comparison costs
    #: microseconds, and it is still far more resolution than "did this person
    #: raise an arm" needs.
    thumb_px: int = 32
    #: Mean absolute difference, normalised to 0..1. Two consecutive frames of
    #: a genuinely static scene sit near zero; sensor and compression noise
    #: land a little above it, which is what this has to clear.
    diff_threshold: float = 0.02
    #: Heartbeat, in analysed frames. See guard 2 in the module docstring.
    max_stale_frames: int = 60
    #: The gate refuses to skip until the model has had this many real looks at
    #: a track. See guard 3 -- this one is not an optimisation knob, it is what
    #: stops the gate starving the voter that depends on it.
    min_runs_before_skip: int = 8

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> ActivityConfig:
        block = dict(cfg.get("activity_gate", {}) or {})
        return cls(
            enabled=bool(block.get("enabled", True)),
            thumb_px=int(block.get("thumb_px", 32)),
            diff_threshold=float(block.get("diff_threshold", 0.02)),
            max_stale_frames=int(block.get("max_stale_frames", 60)),
            min_runs_before_skip=int(block.get("min_runs_before_skip", 8)),
        )


def should_run(
    *,
    first_sighting: bool,
    diff: float,
    frames_since_run: int,
    runs_so_far: int,
    diff_threshold: float,
    max_stale_frames: int,
    min_runs_before_skip: int,
) -> tuple[bool, str]:
    """Decide whether to spend an inference. Pure: numbers in, verdict out.

    Returns ``(run, reason)``. The reason is not decoration -- "why did the
    model not run on the frame where it mattered" is the question anybody
    debugging this will ask first, and a bare boolean cannot answer it.
    """
    if first_sighting:
        # Never skip somebody the first time we see them. A person who walks
        # into frame already holding a weapon and then stands perfectly still
        # would otherwise be gated out forever on their own stillness.
        return True, "first sighting"
    if runs_so_far < min_runs_before_skip:
        # Guard 3. The downstream voter needs a full window of looks before it
        # can confirm anything; skipping during that window means it never
        # confirms, and a still armed person silently never alerts.
        return True, f"warming up ({runs_so_far}/{min_runs_before_skip} looks)"
    if frames_since_run >= max_stale_frames:
        return True, f"heartbeat ({frames_since_run} frames since last run)"
    if diff >= diff_threshold:
        return True, f"motion {diff:.4f} >= {diff_threshold}"
    return False, f"still (motion {diff:.4f} < {diff_threshold})"


def thumbnail_difference(a: Any, b: Any) -> float:
    """Mean absolute difference of two thumbnails, normalised to 0..1.

    Shapes must match; the caller builds both through the same
    ``ActivityGate``, so a mismatch is a bug rather than a runtime condition
    to paper over.
    """
    import numpy as np

    if a.shape != b.shape:
        raise ValueError(f"thumbnail shapes differ: {a.shape} vs {b.shape}")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) / 255.0


@dataclass
class ActivityGate:
    """Per-track thumbnails and run bookkeeping for one second-stage model.

    One gate instance per model per camera: gesture and weapon ask different
    questions and run on their own cadences, so sharing a gate between them
    would let one model's run suppress the other's.
    """

    cfg: ActivityConfig = field(default_factory=ActivityConfig)
    _thumbs: dict[int, Any] = field(default_factory=dict, repr=False)
    _last_run_frame: dict[int, int] = field(default_factory=dict, repr=False)
    _runs_per_track: dict[int, int] = field(default_factory=dict, repr=False)
    #: Counters, surfaced through the health endpoint so the saving is
    #: observable rather than asserted.
    runs: int = 0
    skips: int = 0

    def thumbnail(self, crop: Any) -> Any:
        import cv2

        size = self.cfg.thumb_px
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)

    def check(self, track_id: int, crop: Any, frame_id: int) -> tuple[bool, str]:
        """Should the expensive model run for this track on this frame?

        Records the thumbnail either way, so "changed since we last *looked*"
        stays true even across skipped frames -- comparing only against the
        last frame the model actually ran on would let a slow drift accumulate
        unnoticed, one sub-threshold step at a time.
        """
        if not self.cfg.enabled:
            self.runs += 1
            return True, "gate disabled"

        thumb = self.thumbnail(crop)
        previous = self._thumbs.get(track_id)
        self._thumbs[track_id] = thumb

        diff = 0.0 if previous is None else thumbnail_difference(thumb, previous)
        run, reason = should_run(
            first_sighting=previous is None,
            diff=diff,
            frames_since_run=frame_id - self._last_run_frame.get(track_id, frame_id),
            runs_so_far=self._runs_per_track.get(track_id, 0),
            diff_threshold=self.cfg.diff_threshold,
            max_stale_frames=self.cfg.max_stale_frames,
            min_runs_before_skip=self.cfg.min_runs_before_skip,
        )
        if run:
            self._last_run_frame[track_id] = frame_id
            self._runs_per_track[track_id] = self._runs_per_track.get(track_id, 0) + 1
            self.runs += 1
        else:
            self.skips += 1
        return run, reason

    def close_track(self, track_id: int) -> None:
        self._thumbs.pop(track_id, None)
        self._last_run_frame.pop(track_id, None)
        self._runs_per_track.pop(track_id, None)

    def stats(self) -> dict[str, Any]:
        total = self.runs + self.skips
        return {
            "runs": self.runs,
            "skips": self.skips,
            "skipped_pct": round(self.skips / total * 100, 1) if total else 0.0,
        }
