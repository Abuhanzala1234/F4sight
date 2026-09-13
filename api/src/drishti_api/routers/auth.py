"""Login and token refresh (§8, §12)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import AuditLog, User
from ..schemas import LoginIn, TokenOut, UserOut
from ..security import (
    Principal,
    create_token,
    current_principal,
    decode_token,
    login_limiter,
    verify_password,
)
from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/login", response_model=TokenOut)
async def login(
    payload: LoginIn,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenOut:
    ip = _client_ip(request)
    login_limiter.check(ip)

    user = (
        await db.execute(select(User).where(User.username == payload.username))
    ).scalar_one_or_none()

    # Identical response for "no such user" and "wrong password": distinguishing
    # them hands an attacker a list of valid usernames.
    if user is None or not user.active or not verify_password(payload.password, user.password_hash):
        logger.warning("failed login for username=%r from ip=%s", payload.username, ip)
        db.add(
            AuditLog(
                action="auth.login_failed",
                target_type="user",
                target_id=payload.username,
                ip=ip,
            )
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid username or password",
        )

    login_limiter.reset(ip)
    user.last_login_at = datetime.now(UTC)
    db.add(
        AuditLog(
            actor_id=user.id,
            action="auth.login",
            target_type="user",
            target_id=user.id,
            ip=ip,
        )
    )

    access, ttl = create_token(user.id, user.role, settings)
    refresh, _ = create_token(user.id, user.role, settings, refresh=True)
    return TokenOut(
        access_token=access,
        refresh_token=refresh,
        expires_in=ttl,
        role=user.role,  # type: ignore[arg-type]
        display_name=user.display_name,
    )


@router.post("/refresh", response_model=TokenOut)
async def refresh_token(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenOut:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer refresh token")

    payload = decode_token(header.split(" ", 1)[1], settings)
    if payload.get("typ") != "refresh":
        raise HTTPException(401, "an access token cannot be exchanged for a new one")

    user = (await db.execute(select(User).where(User.id == payload["sub"]))).scalar_one_or_none()
    if user is None or not user.active:
        raise HTTPException(401, "user no longer active")

    access, ttl = create_token(user.id, user.role, settings)
    rotated, _ = create_token(user.id, user.role, settings, refresh=True)
    return TokenOut(
        access_token=access,
        refresh_token=rotated,  # rotate on every use
        expires_in=ttl,
        role=user.role,  # type: ignore[arg-type]
        display_name=user.display_name,
    )


@router.get("/me", response_model=UserOut)
async def me(
    principal: Annotated[Principal, Depends(current_principal)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    user = (await db.execute(select(User).where(User.id == principal.user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(404, "user not found")
    return user
