"""Shared fixtures. We mock the GPU, Fabric and the network — never the logic."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpers import T0, make_detection, make_track

from drishti_worker.types import (
    CameraRuntime,
    ZoneKind,
    ZoneRuntime,
)


@pytest.fixture
def t0() -> datetime:
    return T0


@pytest.fixture
def camera() -> CameraRuntime:
    return CameraRuntime(
        camera_id="cam-1",
        code="CAM-01",
        site_id="site-1",
        site_code="BOP-03",
        timezone="Asia/Kolkata",
        width=1280,
        height=720,
        analytics_fps=6.0,
    )


@pytest.fixture
def area_zone() -> ZoneRuntime:
    return ZoneRuntime(
        zone_id="z-area",
        name="Restricted",
        kind=ZoneKind.AREA,
        polygon=((100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)),
        classes=("person",),
        severity_base=3,
    )


@pytest.fixture
def tripwire_zone() -> ZoneRuntime:
    return ZoneRuntime(
        zone_id="z-wire",
        name="Perimeter",
        kind=ZoneKind.TRIPWIRE,
        # Wound bottom-to-top so that walking left-to-right is INBOUND, i.e.
        # into the protected area. Reversing the two points flips the labels.
        polygon=((600.0, 720.0), (600.0, 0.0)),
        direction="both",
        classes=("person",),
        severity_base=4,
    )


@pytest.fixture
def mask_zone() -> ZoneRuntime:
    return ZoneRuntime(
        zone_id="z-mask",
        name="Public road",
        kind=ZoneKind.MASK,
        polygon=((0.0, 600.0), (1280.0, 600.0), (1280.0, 720.0), (0.0, 720.0)),
    )


@pytest.fixture
def track_factory():
    return make_track


@pytest.fixture
def detection_factory():
    return make_detection
