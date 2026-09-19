"""Alert assembly: snapshot, clip, evidence document, hash, sinks.

This is where §7.11's chain is built, in the worker, at assembly time:

    file bytes -> item SHA-256 -> evidence doc -> canonical bytes -> evidence_hash

and where P4 is enforced in code: the snapshot written here is
``frame.image`` — the ORIGINAL — never the enhanced array the detector saw. The
enhancement parameters ride along as metadata so the record is complete without
being altered.

Clips are pre-roll only, cut synchronously from ``FrameBuffer``. An alert must
reach the operator's screen in under a second (§3.4), which rules out waiting
for ``clip_post_roll_s`` of frames that have not happened yet — see the scope
note on ``_write_clip`` for the reasoning and what a post-roll pass would need.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .evidence import assemble, evidence_hash
from .sinks import AlertRecord, FanoutSink, MinioSink
from .types import CameraRuntime, Frame, RiskResult, Signal, Track

logger = logging.getLogger(__name__)

__all__ = ["AlertAssembler", "FrameBuffer"]


class FrameBuffer:
    """Rolling buffer of recent original frames, for clip pre-roll.

    Bounded by frame count, not seconds, because at 6 fps a 5-second pre-roll is
    30 frames and holding a variable number of 1080p arrays is how a worker's
    RSS surprises you at 3 a.m.
    """

    def __init__(self, max_frames: int = 60) -> None:
        self._frames: deque[Frame] = deque(maxlen=max_frames)

    def push(self, frame: Frame) -> None:
        self._frames.append(frame)

    def window(self, around: datetime, pre_s: float, post_s: float) -> list[Frame]:
        lo = around.timestamp() - pre_s
        hi = around.timestamp() + post_s
        return [f for f in self._frames if lo <= f.ts_utc.timestamp() <= hi]

    def __len__(self) -> int:
        return len(self._frames)


@dataclass
class AlertAssembler:
    """Turns a pipeline alert callback into a persisted, hashed alert."""

    sinks: FanoutSink
    minio: MinioSink | None
    config_version: str
    spec_version: str
    worker_version: str
    snapshot_quality: int = 92
    clip_pre_roll_s: float = 5.0
    clip_post_roll_s: float = 5.0
    clip_fps: int = 8

    def build(
        self,
        *,
        alert_id: str,
        camera: CameraRuntime,
        track: Track,
        frame: Frame,
        signals: Sequence[Signal],
        risk: RiskResult,
        profile: str,
        enhancement_params: Mapping[str, Any],
        escalated: bool = False,
        suppressed_since_last: int = 0,
        frame_buffer: FrameBuffer | None = None,
    ) -> AlertRecord:
        now = datetime.now(UTC)
        items: list[dict[str, Any]] = []

        snapshot = self._write_snapshot(alert_id, camera, frame, enhancement_params)
        if snapshot is not None:
            items.append(snapshot)

        clip = self._write_clip(alert_id, camera, frame, frame_buffer, enhancement_params)
        if clip is not None:
            items.append(clip)

        # THE alert's kind is the STRONGEST signal, not whichever rule
        # happened to run first (RuleEngine evaluates its rules in a fixed
        # registration order, and WeaponVisibleRule is registered well after
        # the zone/tripwire/perimeter rules) -- signals[0] silently picked
        # registration order instead, so a frame where a weapon (weight ~75)
        # fired alongside a zone/perimeter signal (weight ~18-45) reported
        # kind="ZONE_INTRUSION" or "PERIMETER_APPROACH" on the dashboard with
        # "WEAPON_VISIBLE" sitting one slot down in reason_codes where an
        # operator triaging by kind would never see it. Same invariant
        # pipeline.py's own `primary` already follows when it picks which
        # signal drives the debounce key -- this just makes the alert itself
        # agree with what debounce already decided mattered most.
        primary = max(signals, key=lambda s: s.weight)
        doc = assemble(
            alert_id=alert_id,
            site={
                "id": camera.site_id,
                "code": camera.site_code,
                "timezone": camera.timezone,
            },
            camera={
                "id": camera.camera_id,
                "code": camera.code,
                "width": camera.width,
                "height": camera.height,
                "analytics_fps": camera.analytics_fps,
            },
            detection={
                "track_id": track.track_id,
                "class": track.cls,
                "box": [round(v, 2) for v in track.box],
                "foot_point": [round(v, 2) for v in track.foot_point],
                "conf": round(track.conf, 4),
                "max_conf": round(track.max_conf, 4),
                "age_frames": track.age_frames,
                "hits": track.hits,
                "first_seen": track.first_seen.isoformat(),
                "last_seen": track.last_seen.isoformat(),
                "frame_id": frame.frame_id,
                "evqm_profile": profile,
                # P4: what the MODEL saw is recorded as metadata. What was
                # STORED is the original frame.
                "enhancement_params": dict(enhancement_params),
            },
            risk={
                "score": risk.score,
                "severity": risk.severity,
                "breakdown": [
                    {"code": s.code, "weight": s.weight, "detail": dict(s.detail)}
                    for s in risk.breakdown
                ],
                "reason_codes": list(risk.reason_codes()),
            },
            items=items,
            config_version=self.config_version,
            spec_version=self.spec_version,
            worker_version=self.worker_version,
            created_at=now.isoformat(),
        )

        digest = evidence_hash(doc)
        doc_with_hash = {**doc, "evidence_hash": digest}

        record = AlertRecord(
            alert_id=alert_id,
            site_id=camera.site_id,
            site_code=camera.site_code,
            camera_id=camera.camera_id,
            camera_code=camera.code,
            track_id=track.track_id,
            zone_id=primary.detail.get("zone_id"),
            kind=primary.code,
            severity=risk.severity,
            risk_score=risk.score,
            risk_breakdown=[
                {"code": s.code, "weight": s.weight, "detail": dict(s.detail)}
                for s in risk.breakdown
            ],
            reason_codes=list(risk.reason_codes()),
            ts_utc=frame.ts_utc.isoformat(),
            window_start=track.first_seen.isoformat(),
            window_end=track.last_seen.isoformat(),
            evidence_hash=digest,
            evidence_doc=doc_with_hash,
            evidence_items=items,
        )

        logger.info(
            "ALERT camera=%s kind=%s severity=%s risk=%.1f track=%d reasons=%s%s hash=%s…",
            camera.code,
            record.kind,
            record.severity,
            record.risk_score,
            track.track_id,
            ",".join(record.reason_codes),
            f" (escalated, {suppressed_since_last} suppressed)" if escalated else "",
            digest[:12],
        )
        return record

    def emit(self, record: AlertRecord) -> dict[str, bool]:
        return self.sinks.emit(record)

    # -- media -------------------------------------------------------------

    def _write_snapshot(
        self,
        alert_id: str,
        camera: CameraRuntime,
        frame: Frame,
        enhancement_params: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Encode and store the ORIGINAL frame. P4, enforced here."""
        if self.minio is None:
            return None
        try:
            import cv2

            ok, buffer = cv2.imencode(
                ".jpg",
                frame.image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.snapshot_quality],
            )
            if not ok:
                logger.error("camera=%s JPEG encode failed for alert=%s", camera.code, alert_id)
                return None

            key = (
                f"{camera.site_code}/{camera.code}/"
                f"{frame.ts_utc.strftime('%Y/%m/%d')}/{alert_id}-snapshot.jpg"
            )
            stored = self.minio.put("snapshot", key, buffer.tobytes(), "image/jpeg")
            return {
                **stored,
                "id": str(uuid.uuid4()),
                "width": frame.width,
                "height": frame.height,
                "captured_at": frame.ts_utc.isoformat(),
                # Always false for a snapshot. The assembler in evidence.py
                # rejects the document outright if this is ever true.
                "enhanced": False,
                "enhancement_params": dict(enhancement_params),
            }
        except Exception:
            logger.exception(
                "camera=%s snapshot storage failed for alert=%s; "
                "the alert will still fire without it",
                camera.code,
                alert_id,
            )
            return None

    def _write_clip(
        self,
        alert_id: str,
        camera: CameraRuntime,
        frame: Frame,
        frame_buffer: FrameBuffer | None,
        enhancement_params: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Mux the buffered ORIGINAL frames around the alert into an MP4. P4.

        **Scope decision, stated plainly rather than hidden in a corner case:**
        this only covers *pre-roll* — the ``clip_pre_roll_s`` seconds already in
        the buffer at the moment the alert fires. ``clip_post_roll_s`` is kept in
        config for a deployment that adds an async second pass, but is not used
        here. The reason is §3.4: post-roll frames have not happened yet when the
        alert needs to reach the operator, and this assembler's contract is to
        hash the document once, synchronously, at assembly time (§7.11) — hold
        it open for five more seconds of video and either the alert is five
        seconds late or the hash is computed before the clip exists. Pre-roll
        alone still answers the operator's first question, "what led up to
        this", which is most of what a clip is for.

        Bounded and fast on purpose: the buffer already holds only the last
        ``clip_pre_roll_s`` seconds, so this encodes at most a few dozen frames
        — tens of milliseconds, not the "muxing five seconds takes longer than
        the latency budget" case the module docstring warns about (that concern
        applies to a full pre+post clip, which is exactly what this does not
        attempt).
        """
        if self.minio is None or frame_buffer is None or self.clip_pre_roll_s <= 0:
            return None
        frames = frame_buffer.window(frame.ts_utc, self.clip_pre_roll_s, 0.0)
        if len(frames) < 2:
            return None  # a one-frame "clip" is a snapshot with extra steps

        import tempfile
        from pathlib import Path

        try:
            import cv2

            height, width = frames[0].image.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            tmp_path = Path(tempfile.mktemp(suffix=".mp4"))
            writer = cv2.VideoWriter(str(tmp_path), fourcc, self.clip_fps, (width, height))
            try:
                if not writer.isOpened():
                    logger.error(
                        "camera=%s could not open video writer for alert=%s",
                        camera.code,
                        alert_id,
                    )
                    return None
                for buffered in frames:
                    writer.write(buffered.image)
            finally:
                writer.release()

            data = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            if not data:
                logger.error(
                    "camera=%s clip encode produced 0 bytes for alert=%s",
                    camera.code,
                    alert_id,
                )
                return None

            key = (
                f"{camera.site_code}/{camera.code}/"
                f"{frame.ts_utc.strftime('%Y/%m/%d')}/{alert_id}-clip.mp4"
            )
            stored = self.minio.put("clip", key, data, "video/mp4")
            return {
                **stored,
                "id": str(uuid.uuid4()),
                "width": width,
                "height": height,
                "fps": self.clip_fps,
                "frame_count": len(frames),
                "window_start": frames[0].ts_utc.isoformat(),
                "window_end": frames[-1].ts_utc.isoformat(),
                # Always false: these are the same original frames the snapshot
                # comes from, never the enhanced array the detector saw (P4).
                "enhanced": False,
                "enhancement_params": dict(enhancement_params),
            }
        except Exception:
            logger.exception(
                "camera=%s clip storage failed for alert=%s; "
                "the alert will still fire without it",
                camera.code,
                alert_id,
            )
            return None
