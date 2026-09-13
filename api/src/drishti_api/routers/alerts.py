"""Alert feed, detail, adjudication and evidence access (§8). **[DEMO-CRITICAL]**"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Select, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db import get_db
from ..models import Alert, AuditLog, EvidenceItem
from ..schemas import AdjudicateIn, AlertDetail, AlertPage, AlertSummary
from ..security import RequireInvestigator, RequireOperator, RequireViewer
from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])


def _encode_cursor(ts: datetime, alert_id: str) -> str:
    return base64.urlsafe_b64encode(
        json.dumps({"ts": ts.isoformat(), "id": alert_id}).encode()
    ).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return datetime.fromisoformat(payload["ts"]), payload["id"]
    except (ValueError, KeyError, binascii.Error, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"malformed cursor: {exc}") from exc


@router.get("", response_model=AlertPage)
async def list_alerts(
    principal: RequireViewer,
    db: Annotated[AsyncSession, Depends(get_db)],
    site_id: str | None = None,
    camera_id: str | None = None,
    severity: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
) -> AlertPage:
    """Keyset pagination on ``(ts_utc, id)``.

    Not OFFSET: the alert feed is written to constantly, so an offset page two
    silently skips or repeats rows as new alerts arrive. A keyset cursor is
    stable under concurrent writes, which is the normal case here.

    ``from`` is also how a reconnecting dashboard replays what it missed while
    its WebSocket was down (§9) — a dropped socket must never mean a lost alert
    on screen.
    """
    stmt: Select[Any] = select(Alert)

    if site_id:
        stmt = stmt.where(Alert.site_id == site_id)
    if camera_id:
        stmt = stmt.where(Alert.camera_id == camera_id)
    if severity:
        stmt = stmt.where(Alert.severity.in_(severity.split(",")))
    if kind:
        stmt = stmt.where(Alert.kind.in_(kind.split(",")))
    if status:
        stmt = stmt.where(Alert.status == status)
    if from_ts:
        stmt = stmt.where(Alert.ts_utc >= from_ts)
    if to_ts:
        stmt = stmt.where(Alert.ts_utc <= to_ts)
    if cursor:
        cur_ts, cur_id = _decode_cursor(cursor)
        stmt = stmt.where(
            or_(Alert.ts_utc < cur_ts, and_(Alert.ts_utc == cur_ts, Alert.id < cur_id))
        )

    rows = list(
        (
            await db.execute(stmt.order_by(Alert.ts_utc.desc(), Alert.id.desc()).limit(limit + 1))
        ).scalars()
    )

    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = _encode_cursor(page[-1].ts_utc, page[-1].id) if has_more and page else None

    return AlertPage(items=[AlertSummary.model_validate(a) for a in page], next_cursor=next_cursor)


@router.get("/{alert_id}", response_model=AlertDetail)
async def get_alert(
    alert_id: str,
    principal: RequireViewer,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Alert:
    alert = (
        await db.execute(
            select(Alert).options(selectinload(Alert.items)).where(Alert.id == alert_id)
        )
    ).scalar_one_or_none()
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")
    return alert


@router.post("/{alert_id}/ack", response_model=AlertSummary)
async def acknowledge(
    alert_id: str,
    principal: RequireOperator,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Alert:
    alert = (await db.execute(select(Alert).where(Alert.id == alert_id))).scalar_one_or_none()
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")
    if alert.status == "raised":
        alert.status = "acknowledged"
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="alert.ack",
            target_type="alert",
            target_id=alert_id,
        )
    )
    return alert


@router.post("/{alert_id}/adjudicate", response_model=AlertSummary)
async def adjudicate(
    alert_id: str,
    payload: AdjudicateIn,
    principal: RequireOperator,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Alert:
    """The human decides (P7). The system only ever recommended.

    Adjudications feed back into `make eval` as labels, so the measured
    precision improves the more the system is actually used (§15).
    """
    alert = (await db.execute(select(Alert).where(Alert.id == alert_id))).scalar_one_or_none()
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")

    alert.status = "adjudicated"
    alert.adjudication = payload.verdict
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="alert.adjudicate",
            target_type="alert",
            target_id=alert_id,
            detail={"verdict": payload.verdict, "note": payload.note},
        )
    )
    logger.info("alert=%s adjudicated %s by=%s", alert_id, payload.verdict, principal.user_id)
    return alert


@router.get("/{alert_id}/evidence/{item_id}")
async def evidence_url(
    alert_id: str,
    item_id: str,
    principal: RequireInvestigator,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """A short-lived presigned URL for one evidence file.

    Investigator-only, and every access writes an audit row (§12). Evidence
    access is exactly the thing a defence lawyer will ask about, so it is the
    thing we log most carefully.
    """
    item = (
        await db.execute(
            select(EvidenceItem).where(
                EvidenceItem.id == item_id, EvidenceItem.alert_id == alert_id
            )
        )
    ).scalar_one_or_none()
    if item is None:
        raise HTTPException(404, f"evidence item {item_id} not found on alert {alert_id}")

    try:
        from datetime import timedelta

        from minio import Minio

        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        url = client.presigned_get_object(
            item.bucket,
            item.object_key,
            expires=timedelta(seconds=settings.presigned_url_ttl_s),
        )
    except Exception as exc:
        logger.exception("could not presign evidence item=%s", item_id)
        raise HTTPException(503, f"object storage unavailable: {exc}") from exc

    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="evidence.access",
            target_type="evidence_item",
            target_id=item_id,
            detail={"alert_id": alert_id, "kind": item.kind, "sha256": item.sha256},
        )
    )
    return {
        "url": url,
        "expires_in": settings.presigned_url_ttl_s,
        "sha256": item.sha256,
        "kind": item.kind,
        # P4, restated where a downstream consumer will see it.
        "enhanced": item.enhanced,
        "note": "This is the original, unenhanced frame. Verify with the sha256 above.",
    }
