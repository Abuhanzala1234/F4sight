"""Health and metrics (§8, §13).

Health is honest: a degraded system says degraded. A dashboard that shows green
while Redis is down teaches operators to ignore the dashboard.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import __version__
from ..db import get_db
from ..schemas import ComponentHealth, HealthOut
from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


async def _check_db(db: AsyncSession) -> ComponentHealth:
    started = time.monotonic()
    try:
        await db.execute(text("SELECT 1"))
        return ComponentHealth(
            name="postgres",
            ok=True,
            latency_ms=round((time.monotonic() - started) * 1000, 1),
        )
    except Exception as exc:
        logger.exception("postgres health check failed")
        return ComponentHealth(name="postgres", ok=False, detail=str(exc)[:200])


def _check_redis(settings: Settings) -> ComponentHealth:
    started = time.monotonic()
    try:
        import redis

        redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        return ComponentHealth(
            name="redis",
            ok=True,
            latency_ms=round((time.monotonic() - started) * 1000, 1),
        )
    except Exception as exc:
        # Redis down degrades WS fan-out to polling; the DB path is unaffected
        # (§13). Worth reporting, not worth panicking about.
        return ComponentHealth(name="redis", ok=False, detail=str(exc)[:200])


def _check_minio(settings: Settings) -> ComponentHealth:
    try:
        from minio import Minio

        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        exists = client.bucket_exists(settings.minio_bucket_evidence)
        return ComponentHealth(
            name="minio",
            ok=exists,
            detail=("" if exists else f"bucket {settings.minio_bucket_evidence} is missing"),
        )
    except Exception as exc:
        return ComponentHealth(name="minio", ok=False, detail=str(exc)[:200])


def _worker_cameras(settings: Settings) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-camera state, as last reported by the worker (sinks.HealthPublisher).

    None of this is knowable from inside the API: stream state, measured fps,
    the EVQM profile in force and the reconnect count all live in the worker's
    memory, in another process. The key carries a TTL, so its *absence* is
    itself the answer -- a worker that died stops refreshing it and this
    reports the analytics as down rather than serving its last known good
    snapshot forever, which would be the dashboard telling a comfortable lie.
    """
    try:
        import redis

        raw = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).get(
            settings.worker_health_key
        )
    except Exception as exc:
        return [], [f"could not read worker health: {str(exc)[:120]}"]

    if raw is None:
        return [], ["analytics worker is not reporting (no health snapshot); cameras unknown"]

    try:
        snapshot = json.loads(raw)
    except ValueError:
        return [], ["worker health snapshot is unreadable"]

    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return [], ["worker health snapshot carried no camera list"]

    warnings: list[str] = []
    summary: list[dict[str, Any]] = []
    for entry in cameras:
        if not isinstance(entry, dict):
            continue
        ingest = entry.get("ingest") or {}
        state = str(entry.get("state", "unknown"))
        summary.append(
            {
                "camera_id": entry.get("camera_id"),
                "code": entry.get("camera_code"),
                "state": state,
                "evqm_profile": entry.get("evqm_profile"),
                "fps_in": ingest.get("fps_in"),
                "tracks_active": entry.get("tracks_active"),
                "reconnects": ingest.get("reconnects"),
                "last_error": ingest.get("last_error"),
            }
        )
        # A camera the worker is retrying is not an outage, but it is not
        # something to render green either -- say so once, by name.
        if state not in ("live", "StreamState.LIVE"):
            summary[-1]["ok"] = False
            warnings.append(f"camera {entry.get('camera_code') or entry.get('camera_id')}: {state}")
        else:
            summary[-1]["ok"] = True

    return summary, warnings


@router.get("/health", response_model=HealthOut)
async def health(
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthOut:
    components = [await _check_db(db), _check_redis(settings), _check_minio(settings)]
    critical_ok = components[0].ok  # only the database is load-bearing for reads
    all_ok = all(c.ok for c in components)
    cameras, camera_warnings = _worker_cameras(settings)
    return HealthOut(
        status="ok" if all_ok else ("degraded" if critical_ok else "down"),
        version=__version__,
        components=components,
        cameras=cameras,
        warnings=[*settings.warn_on_dev_secrets(), *camera_warnings],
    )


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Process liveness only — touches no dependency, so a database outage does
    not cause an orchestrator to restart a perfectly healthy API."""
    return {"status": "alive", "version": __version__}


@router.get("/metrics")
async def metrics(db: Annotated[AsyncSession, Depends(get_db)]) -> Response:
    """Prometheus text format (§8)."""
    lines = [
        "# HELP ibvap_api_up API process is running",
        "# TYPE ibvap_api_up gauge",
        "ibvap_api_up 1",
    ]
    try:
        rows = (
            await db.execute(
                text(
                    "SELECT severity, count(*) FROM alert "
                    "WHERE ts_utc > now() - interval '24 hours' GROUP BY severity"
                )
            )
        ).all()
        lines += [
            "# HELP ibvap_alerts_24h Alerts raised in the last 24 hours",
            "# TYPE ibvap_alerts_24h gauge",
            *[f'ibvap_alerts_24h{{severity="{sev}"}} {count}' for sev, count in rows],
        ]
        pending = (
            await db.execute(text("SELECT count(*) FROM alert WHERE ledger_status = 'pending'"))
        ).scalar_one()
        lines += [
            "# HELP ibvap_ledger_pending Alerts awaiting a ledger anchor",
            "# TYPE ibvap_ledger_pending gauge",
            f"ibvap_ledger_pending {pending}",
        ]
    except Exception:
        logger.exception("metrics query failed; serving liveness only")
        lines.append("ibvap_metrics_degraded 1")

    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
