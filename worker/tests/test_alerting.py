"""Alert assembly: snapshot, clip, hash (§7.11, §3.4).

The clip is cut from ``FrameBuffer``, so its correctness is a pure function of
what got pushed and when — no MinIO or Postgres needed to test it. We fake the
one boundary that actually talks to a service (``MinioSink.put``) and exercise
everything else for real, including the actual cv2 video encode: a "clip" that
fails to encode is not a clip, and a fake sink would never catch that.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from helpers import T0, make_track

from drishti_worker.alerting import AlertAssembler, FrameBuffer
from drishti_worker.sinks import FanoutSink, NullSink
from drishti_worker.types import CameraRuntime, Frame, RiskResult, Signal

CAMERA = CameraRuntime(
    camera_id="cam-1",
    code="CAM-01",
    site_id="site-1",
    site_code="BOP-03",
    timezone="Asia/Kolkata",
    width=64,
    height=48,
    analytics_fps=6.0,
)


def make_frame(camera_id: str, frame_id: int, ts: datetime) -> Frame:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 0] = frame_id % 255  # distinguishable content per frame
    return Frame(camera_id, frame_id, ts, image, 64, 48)


class FakeMinio:
    """Records every upload instead of talking to a bucket."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes, str]] = []

    def put(self, kind: str, object_key: str, data: bytes, content_type: str) -> dict:
        import hashlib

        self.calls.append((kind, object_key, data, content_type))
        return {
            "kind": kind,
            "bucket": f"drishti-{kind}s",
            "object_key": object_key,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }


def make_assembler(minio, **overrides) -> AlertAssembler:
    return AlertAssembler(
        sinks=FanoutSink(sinks=[NullSink()]),
        minio=minio,
        config_version="cfg-v1",
        spec_version="1.0.0",
        worker_version="1.0.0",
        **overrides,
    )


def build_kwargs(frame_buffer=None, frame=None):
    return {
        "alert_id": "alert-1",
        "camera": CAMERA,
        "track": make_track(),
        "frame": frame or make_frame("cam-1", 10, T0),
        "signals": [Signal("TRIPWIRE_CROSS", 45.0, {"zone_id": "z1"})],
        "risk": RiskResult(45.0, (Signal("TRIPWIRE_CROSS", 45.0, {}),), "high"),
        "profile": "day",
        "enhancement_params": {},
        "frame_buffer": frame_buffer,
    }


class TestFrameBuffer:
    def test_window_selects_frames_in_range(self):
        buf = FrameBuffer(max_frames=60)
        for i in range(10):
            buf.push(make_frame("c", i, T0 + timedelta(seconds=i)))
        window = buf.window(T0 + timedelta(seconds=5), pre_s=2.0, post_s=1.0)
        ids = [f.frame_id for f in window]
        assert ids == [3, 4, 5, 6]

    def test_oldest_frames_are_evicted_past_capacity(self):
        buf = FrameBuffer(max_frames=3)
        for i in range(5):
            buf.push(make_frame("c", i, T0 + timedelta(seconds=i)))
        assert len(buf) == 3
        window = buf.window(T0 + timedelta(seconds=4), pre_s=10.0, post_s=10.0)
        assert [f.frame_id for f in window] == [2, 3, 4]

    def test_empty_buffer_returns_no_frames(self):
        assert FrameBuffer(max_frames=10).window(T0, 5.0, 5.0) == []


class TestSnapshot:
    def test_no_minio_means_no_snapshot_and_no_crash(self):
        assembler = make_assembler(minio=None)
        record = assembler.build(**build_kwargs())
        assert record.evidence_items == []

    def test_snapshot_is_the_original_frame_not_enhanced(self):
        minio = FakeMinio()
        assembler = make_assembler(minio=minio)
        record = assembler.build(**build_kwargs())
        snapshot = next(i for i in record.evidence_items if i["kind"] == "snapshot")
        assert snapshot["enhanced"] is False


class TestClip:
    def test_no_frame_buffer_means_no_clip(self):
        minio = FakeMinio()
        assembler = make_assembler(minio=minio)
        record = assembler.build(**build_kwargs(frame_buffer=None))
        assert not any(i["kind"] == "clip" for i in record.evidence_items)

    def test_single_frame_window_is_not_a_clip(self):
        """One frame is a snapshot with extra steps; do not pretend it's video."""
        minio = FakeMinio()
        assembler = make_assembler(minio=minio, clip_pre_roll_s=5.0)
        buf = FrameBuffer(max_frames=10)
        buf.push(make_frame("cam-1", 10, T0))
        record = assembler.build(**build_kwargs(frame_buffer=buf))
        assert not any(i["kind"] == "clip" for i in record.evidence_items)

    def test_clip_is_encoded_and_uploaded(self):
        minio = FakeMinio()
        assembler = make_assembler(minio=minio, clip_pre_roll_s=5.0, clip_fps=8)
        buf = FrameBuffer(max_frames=60)
        for i in range(20):
            buf.push(make_frame("cam-1", i, T0 + timedelta(seconds=i * 0.25)))
        frame = make_frame("cam-1", 20, T0 + timedelta(seconds=5.0))

        record = assembler.build(**build_kwargs(frame_buffer=buf, frame=frame))

        clip = next(i for i in record.evidence_items if i["kind"] == "clip")
        assert clip["enhanced"] is False
        assert clip["frame_count"] > 1
        assert clip["sha256"]
        upload = next(c for c in minio.calls if c[0] == "clip")
        assert upload[3] == "video/mp4"
        assert len(upload[2]) > 0  # real encoded bytes, not an empty stub

    def test_clip_pre_roll_zero_disables_clip_writing(self):
        minio = FakeMinio()
        assembler = make_assembler(minio=minio, clip_pre_roll_s=0.0)
        buf = FrameBuffer(max_frames=60)
        for i in range(10):
            buf.push(make_frame("cam-1", i, T0 + timedelta(seconds=i)))
        record = assembler.build(**build_kwargs(frame_buffer=buf))
        assert not any(i["kind"] == "clip" for i in record.evidence_items)

    def test_clip_digest_is_inside_the_hashed_document(self):
        """§7.11: an altered clip must invalidate the alert hash."""
        from drishti_worker.evidence import evidence_hash

        minio = FakeMinio()
        assembler = make_assembler(minio=minio, clip_pre_roll_s=5.0)
        buf = FrameBuffer(max_frames=60)
        for i in range(10):
            buf.push(make_frame("cam-1", i, T0 + timedelta(seconds=i * 0.5)))
        frame = make_frame("cam-1", 10, T0 + timedelta(seconds=5.0))
        record = assembler.build(**build_kwargs(frame_buffer=buf, frame=frame))

        assert record.evidence_hash == evidence_hash(record.evidence_doc)
        tampered = dict(record.evidence_doc)
        tampered_items = [dict(i) for i in tampered["items"]]
        for item in tampered_items:
            if item["kind"] == "clip":
                item["sha256"] = "0" * 64
        tampered["items"] = tampered_items
        assert evidence_hash(tampered) != record.evidence_hash


class TestBuildIsFailSoft:
    def test_minio_put_raising_does_not_break_the_alert(self):
        class ExplodingMinio:
            def put(self, *a, **k):
                raise RuntimeError("bucket is on fire")

        assembler = make_assembler(minio=ExplodingMinio(), clip_pre_roll_s=5.0)
        buf = FrameBuffer(max_frames=10)
        for i in range(5):
            buf.push(make_frame("cam-1", i, T0 + timedelta(seconds=i)))
        record = assembler.build(**build_kwargs(frame_buffer=buf))
        # No snapshot, no clip -- but the alert itself was still produced (P8).
        assert record.evidence_items == []
        assert record.alert_id == "alert-1"


@pytest.mark.slow
def test_full_pre_and_post_roll_window_is_not_attempted():
    """Documents the scope decision in ``_write_clip``: only pre-roll is used,
    because post-roll frames have not happened yet at assembly time (§3.4).
    """
    minio = FakeMinio()
    assembler = make_assembler(minio=minio, clip_pre_roll_s=2.0, clip_post_roll_s=5.0)
    buf = FrameBuffer(max_frames=60)
    for i in range(10):
        buf.push(make_frame("cam-1", i, T0 + timedelta(seconds=i * 0.5)))
    frame = make_frame("cam-1", 10, T0 + timedelta(seconds=4.5))
    record = assembler.build(**build_kwargs(frame_buffer=buf, frame=frame))
    clip = next(i for i in record.evidence_items if i["kind"] == "clip")
    # window_end never exceeds the alert frame's own timestamp.
    assert clip["window_end"] <= frame.ts_utc.isoformat()
