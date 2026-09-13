"""Demo seed data — `make seed` (BUILD_SPEC §16 Phase 1).

Creates one site, three cameras, realistic zones, four users spanning the role
ladder, and a small vehicle watchlist. Idempotent: running it twice is a no-op,
so it is safe inside `make demo`.

Zone coordinates are normalised (0..1), as §6.2 requires, so they survive a
camera resolution change.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from drishti_worker.anpr import plate_hmac
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session_factory
from .models import Camera, Site, User, WatchlistVehicle, Zone
from .security import hash_password
from .settings import get_settings

logging.basicConfig(level="INFO", format="%(levelname)-5s %(message)s")
logger = logging.getLogger("seed")

# Demo credentials. Printed at the end so nobody has to grep for them, and
# obviously not for anything but a demo.
DEMO_USERS = [
    ("admin", "Cmdt. A. Sharma", "admin", "drishti-admin"),
    ("operator", "Hav. R. Singh", "operator", "drishti-operator"),
    ("investigator", "Insp. M. Nair", "investigator", "drishti-investigator"),
    ("viewer", "Sector HQ Display", "viewer", "drishti-viewer"),
]

CAMERAS = [
    {
        "code": "CAM-01",
        "name": "Main gate approach",
        "mediamtx_path": "fixture-intrusion",
        "zones": [
            {
                "name": "Perimeter line",
                "kind": "tripwire",
                # Point order sets which side is "in": drawn bottom-to-top so
                # that moving left-to-right is INBOUND, into the post.
                "polygon": [[0.50, 1.0], [0.50, 0.0]],
                "direction": "both",
                "classes": ["person", "vehicle"],
                "severity_base": 4,
            },
            {
                "name": "Restricted apron",
                "kind": "area",
                "polygon": [[0.70, 0.28], [1.0, 0.28], [1.0, 1.0], [0.70, 1.0]],
                "classes": ["person"],
                "severity_base": 5,
            },
            {
                "name": "Public road (ignored)",
                "kind": "mask",
                # A mask is a NEGATIVE zone: detections inside it do not exist.
                # Civilian traffic on the road below the post would otherwise
                # generate an alert a minute (§7.7.2).
                "polygon": [[0.0, 0.86], [1.0, 0.86], [1.0, 1.0], [0.0, 1.0]],
            },
        ],
    },
    {
        "code": "CAM-02",
        "name": "North fence line",
        "mediamtx_path": "fixture-night",
        "zones": [
            {
                "name": "Fence tripwire",
                "kind": "tripwire",
                "polygon": [[0.10, 0.55], [0.95, 0.45]],
                "direction": "in",
                "classes": ["person"],
                "severity_base": 5,
            },
            {
                "name": "Buffer strip",
                "kind": "area",
                "polygon": [[0.10, 0.55], [0.95, 0.45], [0.95, 0.80], [0.10, 0.92]],
                "classes": ["person", "vehicle"],
                "severity_base": 3,
                "schedule": {
                    "tz": "Asia/Kolkata",
                    "windows": [
                        {"days": [0, 1, 2, 3, 4, 5, 6], "from": "18:00", "to": "06:30"}
                    ],
                },
            },
        ],
    },
    {
        "code": "CAM-03",
        "name": "Vehicle checkpoint",
        "mediamtx_path": "fixture-vehicle",
        "zones": [
            {
                "name": "Inspection bay",
                "kind": "area",
                "polygon": [[0.25, 0.40], [0.80, 0.40], [0.80, 0.95], [0.25, 0.95]],
                "classes": ["vehicle"],
                "severity_base": 2,
            }
        ],
    },
]

# Plates go in as plaintext and are stored as HMAC. Nothing here is a real
# registration number.
WATCHLIST_PLATES = [
    ("HR26DA1234", "stolen"),
    ("DL8CAF5030", "watch"),
    ("MH12AB1234", "denied_entry"),
]


async def seed(session: AsyncSession, site_code: str, profile: str) -> None:
    settings = get_settings()

    site = (
        await session.execute(select(Site).where(Site.code == site_code))
    ).scalar_one_or_none()
    if site is None:
        site = Site(
            code=site_code,
            name="Border Out Post 03 — Sector West",
            sector="West",
            lat=28.6139,
            lon=77.2090,
            timezone="Asia/Kolkata",
        )
        session.add(site)
        await session.flush()
        logger.info("created site %s", site_code)
    else:
        logger.info("site %s already exists", site_code)

    users_by_name: dict[str, User] = {}
    for username, display, role, password in DEMO_USERS:
        user = (
            await session.execute(select(User).where(User.username == username))
        ).scalar_one_or_none()
        if user is None:
            user = User(
                username=username,
                display_name=display,
                role=role,
                password_hash=hash_password(password),
            )
            session.add(user)
            await session.flush()
            logger.info("created user %-13s role=%s", username, role)
        users_by_name[username] = user

    rtsp_host = os.getenv("MEDIAMTX_HOST", "localhost")
    rtsp_port = os.getenv("RTSP_PORT", "8554")

    for spec in CAMERAS:
        camera = (
            await session.execute(
                select(Camera).where(
                    Camera.site_id == site.id, Camera.code == spec["code"]
                )
            )
        ).scalar_one_or_none()
        if camera is None:
            camera = Camera(
                site_id=site.id,
                code=spec["code"],
                name=spec["name"],
                mediamtx_path=spec["mediamtx_path"],
                rtsp_url=f"rtsp://{rtsp_host}:{rtsp_port}/{spec['mediamtx_path']}",
                resolution_w=1280,
                resolution_h=720,
                native_fps=25.0,
                analytics_fps=6.0 if profile == "laptop" else 12.0,
                profile=profile,
            )
            session.add(camera)
            await session.flush()
            logger.info("created camera %s (%s)", spec["code"], spec["name"])

        for zone_spec in spec["zones"]:
            zone = (
                await session.execute(
                    select(Zone).where(
                        Zone.camera_id == camera.id, Zone.name == zone_spec["name"]
                    )
                )
            ).scalar_one_or_none()
            if zone is None:
                session.add(
                    Zone(
                        camera_id=camera.id,
                        name=zone_spec["name"],
                        kind=zone_spec["kind"],
                        polygon=zone_spec["polygon"],
                        direction=zone_spec.get("direction"),
                        classes=zone_spec.get("classes"),
                        schedule=zone_spec.get("schedule"),
                        severity_base=zone_spec.get("severity_base", 3),
                    )
                )
                logger.info("  zone %-22s %s", zone_spec["name"], zone_spec["kind"])

    key = settings.plate_hmac_key.encode()
    for plate, category in WATCHLIST_PLATES:
        digest = plate_hmac(plate, key)
        exists = (
            await session.execute(
                select(WatchlistVehicle).where(WatchlistVehicle.plate_hmac == digest)
            )
        ).scalar_one_or_none()
        if exists is None:
            session.add(
                WatchlistVehicle(
                    plate_hmac=digest,
                    category=category,
                    added_by=users_by_name["investigator"].id,
                )
            )
            # Log the HMAC, not the plate. Seeding a watchlist should not put
            # registration numbers into a log file (P6).
            logger.info("  watchlist vehicle %s… (%s)", digest[:12], category)

    await session.commit()


async def main_async(site_code: str, profile: str) -> int:
    factory = get_session_factory()
    try:
        async with factory() as session:
            await seed(session, site_code, profile)
    except Exception as exc:
        logger.error("seeding failed: %s", exc)
        logger.error("is the database up and migrated? try `make up && make migrate`")
        return 1

    print("\n" + "=" * 62)
    print("  DRISHTI-BOP demo users (development credentials)")
    print("=" * 62)
    for username, display, role, password in DEMO_USERS:
        print(f"  {username:<14} {password:<24} {role:<14} {display}")
    print("=" * 62)
    print("  Dashboard: http://localhost:5173   API: http://localhost:8000/api/docs")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed DRISHTI-BOP demo data")
    parser.add_argument("--site", default=os.getenv("DRISHTI_SITE", "BOP-03"))
    parser.add_argument("--profile", default=os.getenv("DRISHTI_PROFILE", "laptop"))
    args = parser.parse_args()
    return asyncio.run(main_async(args.site, args.profile))


if __name__ == "__main__":
    raise SystemExit(main())
