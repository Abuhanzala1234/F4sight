"""Alert assembly: snapshot, clip, evidence document, hash, sinks.

This is where §7.11's chain is built, in the worker, at assembly time:

    file bytes -> item SHA-256 -> evidence doc -> canonical bytes -> evidence_hash

and where P4 is enforced in code: the snapshot written here is
``frame.image`` — the ORIGINAL — never the enhanced array the detector saw. The
enhancement parameters ride along as metadata so the record is complete without
being altered.

Clip writing goes to a small executor. An alert must reach the operator's screen
in under a second (§3.4); muxing five seconds of pre-roll takes longer than that
and has no business on the critical path.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
    clip_executor: ThreadPoolExecutor | None = None

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
    ) -> AlertRecord:
        now = datetime.now(UTC)
        items: list[dict[str, Any]] = []

        snapshot = self._write_snapshot(alert_id, camera, frame, enhancement_params)
        if snapshot is not None:
            items.append(snapshot)

        primary = signals[0]
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
                logger.error(
                    "camera=%s JPEG encode failed for alert=%s", camera.code, alert_id
                )
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
