"""Watchlists (§8, §7.9, §7.10).

Two rules govern this whole module:

* A plate arrives in plaintext, is HMAC'd, and is never echoed back. The table
  has no plaintext column, so there is nothing to leak (P6).
* Person watchlists 403 while face analytics is disabled, which it is by
  default. Enabling faces is a deliberate, audited act — not something you
  discover you have done.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import yaml
from drishti_worker.anpr import plate_hmac, validate_indian_plate
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import AuditLog, WatchlistPerson, WatchlistVehicle
from ..schemas import PersonWatchIn, VehicleWatchIn, VehicleWatchOut
from ..security import RequireAdmin, RequireInvestigator
from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/watchlist", tags=["watchlist"])


def faces_enabled(config_dir: str = "config") -> bool:
    """Read faces.enabled straight from config, not from a cached setting.

    P6 says the default is disabled and that this is the answer to the privacy
    question. Reading the file each time means the answer cannot drift from
    what an auditor sees in the repository.
    """
    path = Path(config_dir) / "faces.yaml"
    if not path.exists():
        return False
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return bool(data.get("faces", {}).get("enabled", False))
    except (OSError, yaml.YAMLError):
        logger.exception("could not read %s; treating face analytics as DISABLED", path)
        return False


@router.get("/vehicles", response_model=list[VehicleWatchOut])
async def list_vehicles(
    principal: RequireInvestigator, db: Annotated[AsyncSession, Depends(get_db)]
) -> list[WatchlistVehicle]:
    """Returns HMACs, never plates. There is no plaintext column to return."""
    return list(
        (
            await db.execute(
                select(WatchlistVehicle)
                .where(WatchlistVehicle.active)
                .order_by(WatchlistVehicle.created_at.desc())
            )
        ).scalars()
    )


@router.post("/vehicles", response_model=VehicleWatchOut, status_code=201)
async def add_vehicle(
    payload: VehicleWatchIn,
    principal: RequireInvestigator,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> WatchlistVehicle:
    valid, normalised = validate_indian_plate(payload.plate)
    if not valid:
        raise HTTPException(
            422,
            f"{payload.plate!r} is not a recognisable Indian registration number "
            f"(normalised to {normalised!r}). Check for OCR confusions.",
        )

    digest = plate_hmac(normalised, settings.plate_hmac_key.encode())
    existing = (
        await db.execute(select(WatchlistVehicle).where(WatchlistVehicle.plate_hmac == digest))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(409, "this plate is already on the watchlist")

    entry = WatchlistVehicle(
        plate_hmac=digest,
        region=payload.region,
        category=payload.category,
        added_by=principal.user_id,
        expires_at=payload.expires_at,
    )
    db.add(entry)
    await db.flush()
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="watchlist.vehicle.add",
            target_type="watchlist_vehicle",
            target_id=entry.id,
            # The audit log records the HMAC too. Auditing a plate by writing it
            # down in plaintext would defeat the point of hashing it.
            detail={"plate_hmac": digest, "category": payload.category},
        )
    )
    logger.info("watchlist vehicle added hmac=%s… by=%s", digest[:12], principal.user_id)
    return entry


@router.delete("/vehicles/{entry_id}", status_code=204)
async def remove_vehicle(
    entry_id: str,
    principal: RequireInvestigator,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    entry = (
        await db.execute(select(WatchlistVehicle).where(WatchlistVehicle.id == entry_id))
    ).scalar_one_or_none()
    if entry is None:
        raise HTTPException(404, "watchlist entry not found")
    entry.active = False
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="watchlist.vehicle.remove",
            target_type="watchlist_vehicle",
            target_id=entry_id,
        )
    )


@router.get("/persons")
async def list_persons(
    principal: RequireAdmin, db: Annotated[AsyncSession, Depends(get_db)]
) -> list[dict[str, object]]:
    if not faces_enabled():
        raise HTTPException(
            403,
            "face analytics is disabled (config/faces.yaml: faces.enabled = false). "
            "This is the default and it is deliberate (Principle P6). Enabling it "
            "is an explicit, audited configuration change.",
        )
    rows = (await db.execute(select(WatchlistPerson).where(WatchlistPerson.active))).scalars()
    return [
        {
            "id": p.id,
            "ref_code": p.ref_code,
            "display_name": p.display_name,
            "category": p.category,
            "expires_at": p.expires_at,
        }
        for p in rows
    ]


@router.post("/persons", status_code=201)
async def add_person(
    payload: PersonWatchIn,
    principal: RequireAdmin,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    """Explicit enrolment is the ONLY path by which a face embedding is ever
    persisted (P6, §7.10). The worker never writes one.
    """
    if not faces_enabled():
        raise HTTPException(
            403,
            "face analytics is disabled (config/faces.yaml). No person may be "
            "enrolled while it is off.",
        )
    person = WatchlistPerson(
        ref_code=payload.ref_code,
        display_name=payload.display_name,
        category=payload.category,
        added_by=principal.user_id,
        expires_at=payload.expires_at,
    )
    db.add(person)
    await db.flush()
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="watchlist.person.enrol",
            target_type="watchlist_person",
            target_id=person.id,
            detail={"ref_code": payload.ref_code},
        )
    )
    logger.warning(
        "FACE ENROLMENT ref_code=%s by=%s — this is an audited privacy-relevant action",
        payload.ref_code,
        principal.user_id,
    )
    return {"id": person.id, "ref_code": person.ref_code}
