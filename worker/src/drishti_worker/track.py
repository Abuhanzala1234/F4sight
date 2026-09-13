"""Multi-object tracking — ByteTrack, reimplemented (BUILD_SPEC §7.5).

Why reimplement rather than pip-install: the algorithm is MIT and about 250
lines, the packaged versions drag in torch, and we need ``update()`` to be pure
with respect to time so tests can run an hour of tracking in a millisecond.

The idea that makes ByteTrack worth using here is the **second association
pass**. Most trackers throw away detections below the confidence threshold.
ByteTrack keeps them and, after matching the confident detections, tries the
leftovers against the tracks that are still unmatched. A person half-occluded by
a fence post drops to 0.2 confidence for a few frames — exactly the moment a
border intrusion is most interesting, and exactly when a single-pass tracker
loses the ID and the rule engine sees a brand-new track that fails
``min_track_age`` and never fires.

``update()`` never calls ``now()``. The timestamp is an argument. This is the
whole reason the tracker is testable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .geometry import iou
from .types import BoxXYXY, Detection, Point, Track

logger = logging.getLogger(__name__)

__all__ = ["ByteTracker", "KalmanBoxFilter", "TrackerConfig"]


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    high_thresh: float = 0.50
    low_thresh: float = 0.10
    match_thresh: float = 0.80  # max IoU *distance* accepted in pass 1
    match_thresh_low: float = 0.50
    match_thresh_unconfirmed: float = 0.70
    max_age: int = 30  # frames a lost track survives
    min_hits: int = 3  # before is_confirmed
    history_len: int = 120

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> TrackerConfig:
        block = dict(cfg.get("tracker", cfg))
        return cls(
            high_thresh=float(block.get("high_thresh", 0.50)),
            low_thresh=float(block.get("low_thresh", 0.10)),
            match_thresh=float(block.get("match_thresh", 0.80)),
            match_thresh_low=float(block.get("match_thresh_low", 0.50)),
            match_thresh_unconfirmed=float(block.get("match_thresh_unconfirmed", 0.70)),
            max_age=int(block.get("max_age", 30)),
            min_hits=int(block.get("min_hits", 3)),
            history_len=int(block.get("history_len", 120)),
        )


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


class KalmanBoxFilter:
    """Constant-velocity filter on ``[cx, cy, aspect, height]`` and velocities.

    Aspect-and-height rather than width-and-height because a person's aspect
    ratio is nearly constant while their pixel height changes smoothly with
    distance; modelling it that way makes the prediction stable when someone
    walks toward the camera.

    Noise scales with height: a 20-px-tall figure at 200 m has far more
    positional uncertainty per frame than a 300-px one at 10 m, and a fixed
    noise term either over-trusts the far one or under-trusts the near one.
    """

    NDIM = 4

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
    ):
        self._std_pos = std_weight_position
        self._std_vel = std_weight_velocity

        self._motion_mat = np.eye(2 * self.NDIM, dtype=np.float64)
        for i in range(self.NDIM):
            self._motion_mat[i, self.NDIM + i] = 1.0
        self._update_mat = np.eye(self.NDIM, 2 * self.NDIM, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.concatenate([measurement, np.zeros(self.NDIM)])
        h = measurement[3]
        std = np.array(
            [
                2 * self._std_pos * h,
                2 * self._std_pos * h,
                1e-2,
                2 * self._std_pos * h,
                10 * self._std_vel * h,
                10 * self._std_vel * h,
                1e-5,
                10 * self._std_vel * h,
            ]
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._std_pos * h,
                self._std_pos * h,
                1e-2,
                self._std_pos * h,
                self._std_vel * h,
                self._std_vel * h,
                1e-5,
                self._std_vel * h,
            ]
        )
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean, cov

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h])
        innovation_cov = np.diag(np.square(std))

        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ cov @ self._update_mat.T + innovation_cov

        # Solve rather than invert: a 4x4 inverse is cheap but solving is both
        # faster and numerically better behaved when the covariance is stiff.
        kalman_gain = np.linalg.solve(proj_cov.T, (cov @ self._update_mat.T).T).T
        innovation = measurement - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = cov - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


def _xyxy_to_xyah(box: BoxXYXY) -> np.ndarray:
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def _xyah_to_xyxy(state: np.ndarray) -> BoxXYXY:
    # cx, cy come straight out of a numpy array, so left uncast they are
    # numpy.float64, not float -- a real subclass of float, so it passes every
    # isinstance(x, float) check downstream, but in numpy 2.x its repr() is
    # "np.float64(123.4)" rather than "123.4". Evidence hashing calls repr()
    # during RFC 8785 number formatting, so an uncast value here silently
    # corrupts every evidence_hash computed from this box: the stored document
    # (serialised by the *standard* JSON encoder, which is numpy-agnostic)
    # looks completely normal, while the hash computed at assembly time is
    # garbage. It matches, and fails verification, for every alert, always.
    # Caught by re-verifying a live alert through /alerts/{id}/verify.
    cx, cy, a, h = (float(v) for v in state[:4])
    h = max(1e-6, h)
    w = max(1e-6, a * h)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------


class _TrackState:
    """Mutable internal track. The public ``Track`` is rebuilt frozen each frame."""

    __slots__ = (
        "age",
        "attributes",
        "cls",
        "conf",
        "cov",
        "first_seen",
        "history",
        "hits",
        "is_activated",
        "last_seen",
        "max_conf",
        "mean",
        "time_since_update",
        "track_id",
    )

    def __init__(
        self,
        track_id: int,
        det: Detection,
        ts: datetime,
        kf: KalmanBoxFilter,
    ) -> None:
        self.track_id = track_id
        self.cls = det.cls
        self.conf = det.conf
        self.max_conf = det.conf
        self.mean, self.cov = kf.initiate(_xyxy_to_xyah(det.box))
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.first_seen = ts
        self.last_seen = ts
        self.history: list[Point] = [det.foot_point]
        self.attributes: dict[str, Any] = dict(det.attributes)
        self.is_activated = False

    @property
    def box(self) -> BoxXYXY:
        return _xyah_to_xyxy(self.mean)

    def to_public(self, min_hits: int) -> Track:
        return Track(
            track_id=self.track_id,
            cls=self.cls,
            box=self.box,
            conf=self.conf,
            max_conf=self.max_conf,
            age_frames=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            history=tuple(self.history),
            attributes=dict(self.attributes),
            min_hits=min_hits,
        )


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def _iou_distance(tracks: Sequence[_TrackState], dets: Sequence[Detection]) -> np.ndarray:
    """Cost matrix of 1 - IoU. Shape (len(tracks), len(dets))."""
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)), dtype=np.float64)
    cost = np.ones((len(tracks), len(dets)), dtype=np.float64)
    for i, trk in enumerate(tracks):
        tbox = trk.box
        for j, det in enumerate(dets):
            # A person box must not inherit a vehicle's track just because the
            # boxes overlap. Class is part of identity.
            if trk.cls != det.cls:
                continue
            cost[i, j] = 1.0 - iou(tbox, det.box)
    return cost


def _linear_assignment(
    cost: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment, rejecting pairs whose cost exceeds ``threshold``."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(rows, cols, strict=True):
        if cost[r, c] <= threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ByteTracker:
    """ByteTrack. One instance per camera.

    ``track_id`` is per-camera and monotonic within a worker run. The database
    keys tracks by ``(camera_id, track_id, first_seen)`` so recycling across a
    restart is harmless.
    """

    def __init__(self, cfg: TrackerConfig | None = None, camera_id: str = "") -> None:
        self.cfg = cfg or TrackerConfig()
        self.camera_id = camera_id
        self._kf = KalmanBoxFilter()
        self._tracked: list[_TrackState] = []
        self._lost: list[_TrackState] = []
        self._next_id = 1
        self._frame_count = 0
        self._just_closed: list[_TrackState] = []

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def active_count(self) -> int:
        return len(self._tracked)

    def update(self, dets: Sequence[Detection], ts: datetime) -> list[Track]:
        """Advance the tracker by one frame. Returns the currently tracked set.

        Pure with respect to time: ``ts`` is an argument and no clock is read.
        """
        self._frame_count += 1
        self._just_closed = []

        high = [d for d in dets if d.conf >= self.cfg.high_thresh]
        low = [d for d in dets if self.cfg.low_thresh <= d.conf < self.cfg.high_thresh]

        # --- predict every existing track forward -------------------------
        pool = self._tracked + self._lost
        for trk in pool:
            trk.mean, trk.cov = self._kf.predict(trk.mean, trk.cov)
            trk.age += 1
            trk.time_since_update += 1

        # --- pass 1: confident detections against all tracks --------------
        cost = _iou_distance(pool, high)
        matches, u_tracks, u_dets_high = _linear_assignment(cost, self.cfg.match_thresh)
        activated: list[_TrackState] = []
        for ti, di in matches:
            self._apply(pool[ti], high[di], ts)
            activated.append(pool[ti])

        # --- pass 2: THE point of ByteTrack. Low-confidence detections get a
        # --- chance against the tracks that pass 1 could not match. This is
        # --- what keeps an ID alive through an occlusion.
        remaining = [pool[i] for i in u_tracks]
        cost_low = _iou_distance(remaining, low)
        matches_low, u_tracks_low, _ = _linear_assignment(cost_low, self.cfg.match_thresh_low)
        for ti, di in matches_low:
            self._apply(remaining[ti], low[di], ts)
            activated.append(remaining[ti])

        still_unmatched = [remaining[i] for i in u_tracks_low]

        # --- births: only from confident detections -----------------------
        for di in u_dets_high:
            det = high[di]
            trk = _TrackState(self._next_id, det, ts, self._kf)
            self._next_id += 1
            activated.append(trk)

        # --- retire what has been missing too long ------------------------
        survivors: list[_TrackState] = []
        for trk in still_unmatched:
            if trk.time_since_update > self.cfg.max_age:
                self._just_closed.append(trk)
            else:
                survivors.append(trk)

        self._tracked = [t for t in activated if t.time_since_update == 0]
        self._lost = survivors + [t for t in activated if t.time_since_update > 0]

        return [t.to_public(self.cfg.min_hits) for t in self._tracked]

    def close_expired(self) -> list[Track]:
        """Tracks that died on the last ``update()``.

        This is not plumbing. It is the hook the face module uses to destroy
        non-matching embeddings (P6, §7.10) and the hook ANPR uses to finalise
        plate voting. Anything holding per-track state must drain it here.
        """
        closed = [t.to_public(self.cfg.min_hits) for t in self._just_closed]
        self._just_closed = []
        return closed

    def all_tracks(self) -> list[Track]:
        return [t.to_public(self.cfg.min_hits) for t in self._tracked + self._lost]

    def reset(self) -> None:
        self._tracked.clear()
        self._lost.clear()
        self._just_closed.clear()
        self._frame_count = 0

    # -- internals ---------------------------------------------------------

    def _apply(self, trk: _TrackState, det: Detection, ts: datetime) -> None:
        trk.mean, trk.cov = self._kf.update(trk.mean, trk.cov, _xyxy_to_xyah(det.box))
        trk.conf = det.conf
        trk.max_conf = max(trk.max_conf, det.conf)
        trk.hits += 1
        trk.time_since_update = 0
        trk.last_seen = ts
        trk.is_activated = True
        trk.history.append(det.foot_point)
        if len(trk.history) > self.cfg.history_len:
            del trk.history[: len(trk.history) - self.cfg.history_len]
        if det.attributes:
            trk.attributes.update(det.attributes)
