"""Alert sinks (BUILD_SPEC §7.13).

Every sink is **fail-soft**: one raising is logged with full context and the
pipeline continues. Losing the Redis publish must not lose the database row, and
neither must stop the next frame from being analysed.

But fail-soft is not the same as fail-silent. From §13: *Postgres down -> alerts
buffered to a local spool, replayed on recovery. **Never drop an alert.*** So a
failing sink spools to disk and the dashboard's health dot goes amber. The
system degrades visibly rather than quietly.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "AlertRecord",
    "AlertSink",
    "FanoutSink",
    "LiveTrackPublisher",
    "MinioSink",
    "NullSink",
    "PostgresSink",
    "RedisSink",
    "SpoolSink",
]


@dataclass(frozen=True, slots=True)
class AlertRecord:
    """A fully assembled alert, ready to persist. Crosses a process boundary,
    so it is serialisable by construction."""

    alert_id: str
    site_id: str
    site_code: str
    camera_id: str
    camera_code: str
    track_id: int
    zone_id: str | None
    kind: str
    severity: str
    risk_score: float
    risk_breakdown: Sequence[Mapping[str, Any]]
    reason_codes: Sequence[str]
    ts_utc: str
    window_start: str
    window_end: str
    evidence_hash: str
    evidence_doc: Mapping[str, Any]
    evidence_items: Sequence[Mapping[str, Any]] = ()
    status: str = "raised"
    ledger_status: str = "pending"

    def as_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "site_id": self.site_id,
            "site_code": self.site_code,
            "camera_id": self.camera_id,
            "camera_code": self.camera_code,
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "kind": self.kind,
            "severity": self.severity,
            "risk_score": self.risk_score,
            "risk_breakdown": list(self.risk_breakdown),
            "reason_codes": list(self.reason_codes),
            "ts_utc": self.ts_utc,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "evidence_hash": self.evidence_hash,
            "evidence_doc": dict(self.evidence_doc),
            "evidence_items": [dict(i) for i in self.evidence_items],
            "status": self.status,
            "ledger_status": self.ledger_status,
        }

    def summary(self) -> dict[str, Any]:
        """The payload pushed over WebSocket — small enough for a live feed."""
        return {
            "alert_id": self.alert_id,
            "site_code": self.site_code,
            "camera_code": self.camera_code,
            "kind": self.kind,
            "severity": self.severity,
            "risk_score": self.risk_score,
            "reason_codes": list(self.reason_codes),
            "ts_utc": self.ts_utc,
            "status": self.status,
        }


@runtime_checkable
class AlertSink(Protocol):
    def emit(self, alert: AlertRecord) -> None: ...

    @property
    def name(self) -> str: ...


class NullSink:
    """For tests. Records what it was given."""

    def __init__(self) -> None:
        self.emitted: list[AlertRecord] = []

    def emit(self, alert: AlertRecord) -> None:
        self.emitted.append(alert)

    @property
    def name(self) -> str:
        return "null"


class SpoolSink:
    """Last-resort durable buffer. Append-only JSONL on local disk.

    When Postgres is unreachable this is what stands between a border incident
    and an alert that never existed. Replayed by ``scripts/replay_spool.py`` on
    recovery.
    """

    def __init__(self, directory: str | Path = "spool") -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, alert: AlertRecord) -> None:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        path = self.dir / f"alerts-{day}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.as_dict(), separators=(",", ":")) + "\n")
            fh.flush()

    @property
    def name(self) -> str:
        return "spool"

    def pending_count(self) -> int:
        return sum(1 for p in self.dir.glob("alerts-*.jsonl") for _ in p.open(encoding="utf-8"))


class PostgresSink:
    """One transaction per alert: the alert row plus its evidence items.

    Both or neither. An alert row pointing at evidence rows that were not
    written is a verification failure waiting to happen.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: Any = None

    def _connection(self) -> Any:
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(self.dsn, min_size=1, max_size=4, open=True)
        return self._pool.connection()

    def emit(self, alert: AlertRecord) -> None:
        from psycopg.types.json import Jsonb

        with self._connection() as conn, conn.transaction():
            conn.execute(
                """
                INSERT INTO alert (
                    id, site_id, camera_id, track_id, zone_id, kind, severity,
                    risk_score, risk_breakdown, reason_codes, ts_utc,
                    window_start, window_end, status, evidence_hash,
                    evidence_doc, ledger_status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    alert.alert_id,
                    alert.site_id,
                    alert.camera_id,
                    alert.track_id,
                    alert.zone_id,
                    alert.kind,
                    alert.severity,
                    alert.risk_score,
                    Jsonb(list(alert.risk_breakdown)),
                    list(alert.reason_codes),
                    alert.ts_utc,
                    alert.window_start,
                    alert.window_end,
                    alert.status,
                    alert.evidence_hash,
                    Jsonb(dict(alert.evidence_doc)),
                    alert.ledger_status,
                ),
            )
            for item in alert.evidence_items:
                conn.execute(
                    """
                    INSERT INTO evidence_item (
                        id, alert_id, kind, bucket, object_key, sha256, bytes,
                        width, height, captured_at, enhanced, enhancement_params
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        item["id"],
                        alert.alert_id,
                        item["kind"],
                        item["bucket"],
                        item["object_key"],
                        item["sha256"],
                        item.get("bytes"),
                        item.get("width"),
                        item.get("height"),
                        item.get("captured_at"),
                        item.get("enhanced", False),
                        Jsonb(item.get("enhancement_params") or {}),
                    ),
                )

    @property
    def name(self) -> str:
        return "postgres"


class RedisSink:
    """Publishes to a capped Redis Stream for API WebSocket fan-out (§9)."""

    def __init__(self, url: str, stream: str = "drishti:alerts", maxlen: int = 10_000) -> None:
        self.url = url
        self.stream = stream
        self.maxlen = maxlen
        self._client: Any = None

    def _redis(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def emit(self, alert: AlertRecord) -> None:
        self._redis().xadd(
            self.stream,
            {"payload": json.dumps(alert.summary(), separators=(",", ":"))},
            maxlen=self.maxlen,
            approximate=True,
        )

    @property
    def name(self) -> str:
        return "redis"


class LiveTrackPublisher:
    """Publishes per-frame track snapshots for the dashboard's live overlay.

    Deliberately NOT an ``AlertSink`` and nothing like one: this is a
    best-effort, high-frequency, ephemeral feed for drawing real boxes over
    the live camera wall, not an evidentiary record. Losing one of these
    messages is invisible to an operator (the next frame arrives in well
    under a second); losing an alert is not, which is why RedisSink above has
    a completely separate, durable path. A short ``maxlen`` on purpose --
    nothing older than a few seconds is ever useful here, unlike the alert
    stream's history replay.
    """

    def __init__(self, url: str, stream: str = "drishti:live", maxlen: int = 500) -> None:
        self.url = url
        self.stream = stream
        self.maxlen = maxlen
        self._client: Any = None

    def _redis(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def publish(self, camera_id: str, ts_utc: datetime, tracks: list[dict[str, Any]]) -> None:
        payload = {"camera_id": camera_id, "ts": ts_utc.isoformat(), "tracks": tracks}
        try:
            self._redis().xadd(
                self.stream,
                {"payload": json.dumps(payload, separators=(",", ":"))},
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception:
            # Fail-soft and quiet on purpose (unlike every AlertSink): this is
            # a cosmetic real-time feed, not evidence, and it fires many times
            # a second per camera -- logging every Redis hiccup at normal
            # levels here would drown out messages that actually matter.
            logger.debug("live track publish failed (non-fatal)", exc_info=True)

    @property
    def name(self) -> str:
        return "live-tracks"


class MinioSink:
    """Uploads evidence media to MinIO and returns keys + digests.

    Not an ``AlertSink``: it runs during evidence assembly, before the alert
    record exists, because each file's SHA-256 has to be inside the hashed
    document (§7.11).
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        *,
        secure: bool = False,
        bucket_evidence: str = "drishti-evidence",
        bucket_clips: str = "drishti-clips",
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.bucket_evidence = bucket_evidence
        self.bucket_clips = bucket_clips
        self._client: Any = None

    def _minio(self) -> Any:
        if self._client is None:
            from minio import Minio

            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
        return self._client

    def put(self, kind: str, object_key: str, data: bytes, content_type: str) -> dict[str, Any]:
        import io
        from hashlib import sha256

        bucket = self.bucket_clips if kind == "clip" else self.bucket_evidence
        digest = sha256(data).hexdigest()
        self._minio().put_object(
            bucket,
            object_key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return {
            "kind": kind,
            "bucket": bucket,
            "object_key": object_key,
            "sha256": digest,
            "bytes": len(data),
        }

    @property
    def name(self) -> str:
        return "minio"


@dataclass
class FanoutSink:
    """Emits to every configured sink, isolating failures.

    The order matters: durable sinks first. If Postgres succeeds and Redis
    fails, the alert exists and the operator sees it a beat later on the next
    poll. If Postgres fails, we spool before even trying Redis.
    """

    sinks: Sequence[AlertSink]
    spool: SpoolSink | None = None
    fail_soft: bool = True
    failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def emit(self, alert: AlertRecord) -> dict[str, bool]:
        results: dict[str, bool] = {}
        durable_ok = False

        for sink in self.sinks:
            try:
                sink.emit(alert)
                results[sink.name] = True
                if sink.name in ("postgres", "spool"):
                    durable_ok = True
            except Exception:
                self.failures[sink.name] += 1
                results[sink.name] = False
                # No silent failures. Full context, every time.
                logger.exception(
                    "sink=%s failed for alert=%s camera=%s kind=%s " "(failure #%d for this sink)",
                    sink.name,
                    alert.alert_id,
                    alert.camera_code,
                    alert.kind,
                    self.failures[sink.name],
                )
                if not self.fail_soft:
                    raise

        if not durable_ok and self.spool is not None:
            try:
                self.spool.emit(alert)
                results["spool"] = True
                logger.warning(
                    "alert=%s spooled to disk because no durable sink accepted it; "
                    "replay with scripts/replay_spool.py once the database is back",
                    alert.alert_id,
                )
            except Exception:
                # The one place we cannot recover. Scream.
                logger.critical(
                    "alert=%s could not be persisted ANYWHERE, including the spool. "
                    "This alert is lost. Check disk space and permissions.",
                    alert.alert_id,
                )
                results["spool"] = False

        return results

    @property
    def name(self) -> str:
        return "fanout"

    def health(self) -> dict[str, Any]:
        return {
            "sinks": [s.name for s in self.sinks],
            "failures": dict(self.failures),
            "degraded": bool(self.failures),
        }
