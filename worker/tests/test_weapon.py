"""Weapon detection (§7.7). The false-positive discipline is the whole point.

An armed-person alert that cries wolf teaches an operator to dismiss it, and
then they dismiss the real one too (P3). These tests are mostly about what must
NOT fire.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibvap_worker.detect.onnx_weapon import decode_weapon_output, nms
from ibvap_worker.weapon import (
    WEAPON_CLASSES,
    WeaponCandidate,
    WeaponConfig,
    WeaponVoter,
    vote_weapon,
)

CHANNELS = 4 + len(WEAPON_CLASSES)


def raw_with(dets: list[tuple[int, float, tuple[float, float, float, float]]], n: int = 20):
    """Build a YOLOv8-shaped [1, 6, N] tensor with the given detections."""
    out = np.zeros((1, CHANNELS, n), dtype=np.float32)
    for i, (cls_id, conf, (cx, cy, w, h)) in enumerate(dets):
        out[0, 0, i], out[0, 1, i], out[0, 2, i], out[0, 3, i] = cx, cy, w, h
        out[0, 4 + cls_id, i] = conf
    return out


class TestNms:
    def test_suppresses_an_overlapping_duplicate(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        assert nms(boxes, scores, 0.45) == [0]

    def test_keeps_disjoint_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        assert sorted(nms(boxes, scores, 0.45)) == [0, 1]

    def test_empty(self):
        assert nms(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), 0.5) == []


class TestDecode:
    def test_finds_the_strongest_weapon(self):
        raw = raw_with([(0, 0.91, (100, 100, 20, 40)), (1, 0.62, (300, 300, 20, 40))])
        got = decode_weapon_output(raw, min_conf=0.55)
        assert got is not None
        assert got.cls == "guns"
        assert got.conf == pytest.approx(0.91, abs=1e-5)
        # box is (cx-w/2, cy-h/2, cx+w/2, cy+h/2) for the WINNING detection
        # (guns, 0.91), not the lower-scoring one -- catches a max() that
        # picked the right score but a mismatched box.
        assert got.box == pytest.approx((90.0, 80.0, 110.0, 120.0), abs=1e-3)

    def test_below_threshold_is_nothing(self):
        raw = raw_with([(0, 0.40, (100, 100, 20, 40))])
        assert decode_weapon_output(raw, min_conf=0.55) is None

    def test_blank_output_is_nothing(self):
        assert decode_weapon_output(np.zeros((1, CHANNELS, 30), dtype=np.float32), 0.55) is None

    def test_returns_plain_python_types(self):
        """numpy scalars reaching the evidence canonicaliser is a bug this
        project has already paid for once (README)."""
        got = decode_weapon_output(raw_with([(1, 0.8, (10, 10, 5, 5))]), 0.55)
        assert type(got.conf) is float
        assert type(got.cls) is str
        assert all(type(v) is float for v in got.box)

    def test_wrong_channel_count_fails_loudly(self):
        """A swapped-in model with a different class count must not silently
        relabel knives as guns."""
        with pytest.raises(ValueError, match="not the weapon model"):
            decode_weapon_output(np.zeros((1, 9, 20), dtype=np.float32), 0.55)

    def test_wrong_rank_fails_loudly(self):
        with pytest.raises(ValueError):
            decode_weapon_output(np.zeros((CHANNELS, 20), dtype=np.float32), 0.55)


class TestVoting:
    def test_a_single_frame_is_not_an_arrest(self):
        """The umbrella-at-forty-metres case: one frame must never settle."""
        assert vote_weapon([WeaponCandidate("guns", 0.9)], min_frames_agreed=3) is None

    def test_settles_once_enough_frames_agree(self):
        got = vote_weapon([WeaponCandidate("guns", 0.9)] * 3, min_frames_agreed=3)
        assert got is not None
        assert got.cls == "guns"
        assert got.frames_agreed == 3

    def test_settled_box_is_the_most_recent_sighting_not_an_average(self):
        seq = [
            WeaponCandidate("guns", 0.9, box=(0.0, 0.0, 10.0, 10.0)),
            WeaponCandidate("guns", 0.9, box=(50.0, 50.0, 60.0, 60.0)),
        ]
        got = vote_weapon(seq, min_frames_agreed=2)
        assert got is not None
        assert got.box == (50.0, 50.0, 60.0, 60.0)

    def test_quiet_frames_do_not_count(self):
        seq = [WeaponCandidate("guns", 0.9), None, WeaponCandidate("guns", 0.9), None]
        assert vote_weapon(seq, min_frames_agreed=3) is None

    def test_a_gun_knife_split_still_settles_as_armed(self):
        """THE case that justifies voting on ARMED rather than on type: a person
        the model keeps reclassifying is still unambiguously holding something.
        Plurality voting would tie here and never fire."""
        seq = [
            WeaponCandidate("guns", 0.8),
            WeaponCandidate("knife", 0.8),
            WeaponCandidate("guns", 0.8),
            WeaponCandidate("knife", 0.8),
        ]
        got = vote_weapon(seq, min_frames_agreed=3)
        assert got is not None
        assert got.frames_agreed == 4
        assert got.cls in WEAPON_CLASSES

    def test_reports_mean_confidence(self):
        got = vote_weapon(
            [
                WeaponCandidate("knife", 0.6),
                WeaponCandidate("knife", 0.8),
                WeaponCandidate("knife", 0.7),
            ],
            min_frames_agreed=3,
        )
        assert got.conf == pytest.approx(0.7, abs=1e-6)

    def test_rejects_nonsense_threshold(self):
        with pytest.raises(ValueError):
            vote_weapon([WeaponCandidate("guns", 1.0)], min_frames_agreed=0)


class TestWeaponVoter:
    def test_holding_a_weapon_settles(self):
        voter = WeaponVoter(min_frames_agreed=3, window_frames=8)
        assert voter.add(WeaponCandidate("guns", 0.9)) is None
        assert voter.add(WeaponCandidate("guns", 0.9)) is None
        assert voter.add(WeaponCandidate("guns", 0.9)) is not None

    def test_putting_it_down_un_settles(self):
        """A weapon is not a permanent fact about a person, unlike a plate."""
        voter = WeaponVoter(min_frames_agreed=3, window_frames=4)
        for _ in range(3):
            voter.add(WeaponCandidate("guns", 0.9))
        last = None
        for _ in range(4):
            last = voter.add(None)
        assert last is None

    def test_reset(self):
        voter = WeaponVoter(min_frames_agreed=2, window_frames=8)
        voter.add(WeaponCandidate("guns", 0.9))
        voter.reset()
        assert voter.add(WeaponCandidate("guns", 0.9)) is None


class TestSharedDetectorConcurrency:
    def test_two_cameras_never_run_on_each_others_crop(self):
        """One OnnxWeaponDetector serves every camera's stage thread and
        reuses one input buffer. Unlocked, camera B's crop overwrote that
        buffer while camera A's run was still reading it -- A's weapon check
        then ran on B's pixels. The fake session fails the test if its input
        changes mid-run or is ever a mix of two crops."""
        import threading
        import time
        from contextlib import nullcontext

        from ibvap_worker.detect import DetectorConfig
        from ibvap_worker.detect.onnx_weapon import OnnxWeaponDetector

        corrupted: list[str] = []

        class SlowFakeSession:
            def run(self, _outputs, feeds):
                buf = next(iter(feeds.values()))
                before = buf.copy()
                time.sleep(0.002)
                if not np.array_equal(before, buf):
                    corrupted.append("changed mid-run")
                elif before.min() != before.max():
                    corrupted.append("mixed crops")
                return [np.zeros((1, CHANNELS, 10), dtype=np.float32)]

        det = OnnxWeaponDetector.__new__(OnnxWeaponDetector)
        det.cfg = DetectorConfig(input_size=(64, 64))
        det.min_conf, det.nms_iou = 0.75, 0.45
        det._session, det._input_name, det._buffer = SlowFakeSession(), "images", None
        det._buffer_lock, det._run_lock = threading.Lock(), nullcontext()

        def camera(value: int) -> None:
            crop = np.full((64, 64, 3), value, dtype=np.uint8)
            for _ in range(40):
                det.detect(crop)

        threads = [threading.Thread(target=camera, args=(v,)) for v in (0, 255)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert corrupted == []


class TestConfig:
    def test_threshold_is_stricter_than_the_detector_default(self):
        """0.45 is the primary detector's floor; weapons must be stricter."""
        assert WeaponConfig().min_conf > 0.45

    def test_from_mapping_reads_the_block(self):
        cfg = WeaponConfig.from_mapping(
            {
                "weapons": {
                    "enabled": True,
                    "min_conf": 0.7,
                    "every_n_frames": 5,
                    "voting": {"min_frames_agreed": 4, "window_frames": 10},
                }
            }
        )
        assert cfg.enabled and cfg.min_conf == 0.7
        assert cfg.every_n_frames == 5
        assert cfg.min_frames_agreed == 4 and cfg.window_frames == 10

    def test_every_n_frames_is_never_zero(self):
        """A zero would make `frame_id % n` raise on the hot path."""
        assert WeaponConfig.from_mapping({"weapons": {"every_n_frames": 0}}).every_n_frames == 1
