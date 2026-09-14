"""Sites, cameras, zones and stream URLs (§8)."""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import AuditLog, Camera, Site, Zone
from ..schemas import CameraConnectIn, CameraOut, SiteOut, StreamOut, ZoneIn, ZoneOut
from ..security import RequireAdmin, RequireOperator, RequireViewer
from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cameras"])


@router.get("/sites", response_model=list[SiteOut])
async def list_sites(
    principal: RequireViewer, db: Annotated[AsyncSession, Depends(get_db)]
) -> list[Site]:
    return list((await db.execute(select(Site).order_by(Site.code))).scalars())


@router.get("/cameras", response_model=list[CameraOut])
async def list_cameras(
    principal: RequireViewer,
    db: Annotated[AsyncSession, Depends(get_db)],
    site_id: str | None = None,
) -> list[Camera]:
    stmt = select(Camera).order_by(Camera.code)
    if site_id:
        stmt = stmt.where(Camera.site_id == site_id)
    return list((await db.execute(stmt)).scalars())


@router.get("/cameras/{camera_id}/stream", response_model=StreamOut)
async def camera_stream(
    camera_id: str,
    principal: RequireViewer,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamOut:
    """HLS and WebRTC URLs for a camera.

    Never the RTSP URL: it carries camera credentials, and no browser can play
    it anyway (blocker #4). Video reaches the dashboard as MediaMTX → HLS.
    """
    camera = (await db.execute(select(Camera).where(Camera.id == camera_id))).scalar_one_or_none()
    if camera is None:
        raise HTTPException(404, f"camera {camera_id} not found")

    return StreamOut(
        camera_id=camera.id,
        mediamtx_path=camera.mediamtx_path,
        hls_url=f"{settings.hls_base}/{camera.mediamtx_path}/index.m3u8",
        webrtc_url=f"http://{settings.mediamtx_host}:{settings.webrtc_port}/"
        f"{camera.mediamtx_path}/whep",
    )


@router.post("/cameras/{camera_id}/connect", response_model=CameraOut)
async def connect_camera(
    camera_id: str,
    payload: CameraConnectIn,
    principal: RequireOperator,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Camera:
    """Bind a camera slot to a live IP camera (e.g. a phone running 'IP
    Webcam') by address alone — click a tile, type an IP, it is live.

    No model or detector step is involved: the worker's detector is already
    loaded once and shared across every camera (§7.4). This only wires up a
    video source — it registers the RTSP pull on the already-running
    MediaMTX instance via its control API (no restart, no YAML edit) and
    flips the camera to enabled. The worker notices the newly-enabled camera
    on its own periodic re-check of the camera table and hot-starts it
    without a restart either.
    """
    camera = (await db.execute(select(Camera).where(Camera.id == camera_id))).scalar_one_or_none()
    if camera is None:
        raise HTTPException(404, f"camera {camera_id} not found")

    # The three seeded demo cameras sit on `fixture-*` paths that the
    # `fixture-streamer` sidecar (docker-compose.yml) publishes into forever,
    # on a 2s retry loop, with no awareness of this camera's enabled state.
    # Repurposing that same path for a real camera loses the race against
    # that sidecar every time -- the tile keeps showing the dummy footage no
    # matter what source this call configures. Give a real camera its own
    # path instead of fighting over a fixture one.
    target_path = camera.mediamtx_path
    if target_path.startswith("fixture-"):
        target_path = f"live-{camera.code.lower()}"

    source = f"rtsp://{payload.ip}:{payload.port}/{payload.path.lstrip('/')}"
    mediamtx_api = f"http://{settings.mediamtx_host}:{settings.mediamtx_api_port}"

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.post(
                f"{mediamtx_api}/v3/config/paths/replace/{target_path}",
                json={"source": source, "rtspTransport": "tcp", "sourceOnDemand": False},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"could not reach mediamtx: {exc}") from exc
    if resp.status_code == 401:
        # MediaMTX's default authInternalUsers scopes unauthenticated `api`
        # access to 127.0.0.1/::1 only. That "localhost" is MediaMTX's own
        # loopback, not this API container's -- a cross-container call is
        # never that address, so this 401 fires every time regardless of the
        # camera's IP. See infra/mediamtx.yml (authInternalUsers override).
        raise HTTPException(
            502,
            "mediamtx refused the control-API call (401) -- its API is only open to "
            "127.0.0.1 by default and this call comes from another container. "
            "Add an authInternalUsers entry (or authMethod: none) for the api action "
            "in infra/mediamtx.yml.",
        )
    if resp.status_code >= 300:
        raise HTTPException(502, f"mediamtx rejected the camera source: {resp.text}")

    camera.mediamtx_path = target_path
    camera.rtsp_url = f"rtsp://{settings.mediamtx_host}:{settings.rtsp_port}/{target_path}"
    camera.enabled = True
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="camera.connect",
            target_type="camera",
            target_id=camera.id,
            detail={"code": camera.code, "ip": payload.ip},
        )
    )
    logger.info("camera=%s connected to ip=%s by=%s", camera.code, payload.ip, principal.user_id)
    await db.flush()
    return camera


@router.post("/cameras/{camera_id}/disconnect", response_model=CameraOut)
async def disconnect_camera(
    camera_id: str,
    principal: RequireOperator,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Camera:
    """The reverse of ``connect_camera``: tear down the MediaMTX source and
    flip the camera back to an empty slot. The worker's reload loop notices
    within its next poll and stops that camera's threads -- no restart, and
    no other camera is touched."""
    camera = (await db.execute(select(Camera).where(Camera.id == camera_id))).scalar_one_or_none()
    if camera is None:
        raise HTTPException(404, f"camera {camera_id} not found")

    mediamtx_api = f"http://{settings.mediamtx_host}:{settings.mediamtx_api_port}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            await client.delete(f"{mediamtx_api}/v3/config/paths/delete/{camera.mediamtx_path}")
        except httpx.HTTPError as exc:
            # Best-effort: the camera row still gets marked disconnected even
            # if MediaMTX is briefly unreachable -- an operator disconnecting
            # a misbehaving camera should never be blocked by the media layer.
            logger.warning("could not remove mediamtx path for camera=%s: %s", camera.code, exc)

    camera.enabled = False
    camera.rtsp_url = None
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="camera.disconnect",
            target_type="camera",
            target_id=camera.id,
            detail={"code": camera.code},
        )
    )
    logger.info("camera=%s disconnected by=%s", camera.code, principal.user_id)
    await db.flush()
    return camera


@router.get("/cameras/{camera_id}/zones", response_model=list[ZoneOut])
async def list_zones(
    camera_id: str,
    principal: RequireViewer,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Zone]:
    return list((await db.execute(select(Zone).where(Zone.camera_id == camera_id))).scalars())


@router.post("/cameras/{camera_id}/zones", response_model=ZoneOut, status_code=201)
async def create_zone(
    camera_id: str,
    payload: ZoneIn,
    principal: RequireAdmin,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Zone:
    """Zones are admin-only: a zone is a policy decision about what counts as
    an intrusion, and it changes what the system alerts on."""
    camera = (await db.execute(select(Camera).where(Camera.id == camera_id))).scalar_one_or_none()
    if camera is None:
        raise HTTPException(404, f"camera {camera_id} not found")

    zone = Zone(
        camera_id=camera_id,
        name=payload.name,
        kind=payload.kind,
        polygon=[list(p) for p in payload.polygon],
        direction=payload.direction,
        classes=payload.classes or None,
        schedule=payload.schedule,
        severity_base=payload.severity_base,
        enabled=payload.enabled,
    )
    db.add(zone)
    await db.flush()
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="zone.create",
            target_type="zone",
            target_id=zone.id,
            detail={"camera_id": camera_id, "kind": payload.kind, "name": payload.name},
        )
    )
    logger.info(
        "zone created camera=%s kind=%s name=%r by=%s",
        camera.code,
        payload.kind,
        payload.name,
        principal.user_id,
    )
    return zone


@router.put("/cameras/{camera_id}/zones/{zone_id}", response_model=ZoneOut)
async def update_zone(
    camera_id: str,
    zone_id: str,
    payload: ZoneIn,
    principal: RequireAdmin,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Zone:
    zone = (
        await db.execute(select(Zone).where(Zone.id == zone_id, Zone.camera_id == camera_id))
    ).scalar_one_or_none()
    if zone is None:
        raise HTTPException(404, f"zone {zone_id} not found on camera {camera_id}")

    zone.name = payload.name
    zone.kind = payload.kind
    zone.polygon = [list(p) for p in payload.polygon]
    zone.direction = payload.direction
    zone.classes = payload.classes or None
    zone.schedule = payload.schedule
    zone.severity_base = payload.severity_base
    zone.enabled = payload.enabled

    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="zone.update",
            target_type="zone",
            target_id=zone_id,
        )
    )
    return zone


@router.delete("/cameras/{camera_id}/zones/{zone_id}", status_code=204)
async def delete_zone(
    camera_id: str,
    zone_id: str,
    principal: RequireAdmin,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    zone = (
        await db.execute(select(Zone).where(Zone.id == zone_id, Zone.camera_id == camera_id))
    ).scalar_one_or_none()
    if zone is None:
        raise HTTPException(404, f"zone {zone_id} not found")
    await db.delete(zone)
    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="zone.delete",
            target_type="zone",
            target_id=zone_id,
        )
    )
