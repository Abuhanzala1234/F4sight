"""FastAPI application (BUILD_SPEC §8).

Async throughout. The worker's hot path is threaded and the two models are not
mixed (CLAUDE.md/Code).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .db import dispose_engine
from .routers import alerts, auth, cameras, health, verify, watchlist, ws
from .settings import get_settings

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    for warning in settings.warn_on_dev_secrets():
        # Loud about insecure defaults. A demo on dev secrets is fine; a BOP on
        # them is not, and nobody should have to read the source to find out.
        logger.warning("SECURITY: %s — fine for a demo, not for deployment", warning)
    logger.info("DRISHTI-BOP API %s starting", __version__)
    yield
    await dispose_engine()
    logger.info("DRISHTI-BOP API stopped")


app = FastAPI(
    title="DRISHTI-BOP API",
    version=__version__,
    description=(
        "AI video analytics for border surveillance on existing CCTV.\n\n"
        "SIH 2026 · PS 26187 · Team SW-73 (ByteForge).\n\n"
        "Every model and service behind this API is free and self-hosted; "
        "see docs/MODELS.md."
    ),
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """A ValueError escaping a handler is a broken invariant, not a bad request.

    The commonest source is the AlertDetail validator refusing to serve a risk
    breakdown that does not sum to its score (P2). Better a loud 500 than a
    confident wrong number on an operator's screen.
    """
    logger.exception("invariant violation serving %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal invariant violation",
            "error": str(exc),
            "hint": "this usually means stored data violates a documented invariant",
        },
    )


for router in (
    health.router,
    auth.router,
    cameras.router,
    alerts.router,
    verify.router,
    watchlist.router,
):
    app.include_router(router, prefix="/api/v1")

app.include_router(ws.router)  # WebSocket paths are not versioned


@app.get("/api", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "name": "DRISHTI-BOP",
        "version": __version__,
        "problem_statement": "SIH 2026 · PS 26187",
        "team": "SW-73 (ByteForge)",
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }


# In production the dashboard is served by the API; in dev, Vite serves it.
# This mount MUST be the exact path "/" and MUST be registered after every
# other route in this file (StaticFiles with html=True serves index.html at
# "/" itself) -- an app.get("/") route registered anywhere above this line
# would win the exact "/" match ahead of the mount and the browser would get
# raw JSON at the dashboard's own URL instead of the app. That exact bug
# shipped once already: the info route above used to be at "/", not "/api".
_dashboard = Path(__file__).resolve().parents[3] / "dashboard" / "dist"
if _dashboard.exists():
    app.mount("/", StaticFiles(directory=str(_dashboard), html=True), name="dashboard")
    logger.info("serving the built dashboard from %s", _dashboard)
