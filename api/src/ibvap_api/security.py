"""Authentication and RBAC (BUILD_SPEC §8, §12).

Roles are a strict ladder:

    viewer < operator < investigator < admin

``viewer`` sees alerts but no media. ``operator`` adds acknowledge/adjudicate
and live video. ``investigator`` adds evidence download and plate reveal.
``admin`` adds config, watchlists and users.

RBAC is enforced by a dependency, never inside a router body (§8). A check
written in the handler is a check somebody forgets to write in the next handler.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .settings import Settings, get_settings

logger = logging.getLogger(__name__)

ROLE_ORDER = ("viewer", "operator", "investigator", "admin")
_hasher = PasswordHasher()
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        # A malformed hash in the database is a real problem and must be seen.
        logger.exception("password hash verification failed unexpectedly")
        return False


def create_token(
    subject: str, role: str, settings: Settings, *, refresh: bool = False
) -> tuple[str, int]:
    ttl = settings.refresh_token_ttl_s if refresh else settings.access_token_ttl_s
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "role": role,
        "typ": "refresh" if refresh else "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl)).timestamp()),
    }
    return (
        jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm),
        ttl,
    )


def decode_token(token: str, settings: Settings) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


class LoginRateLimiter:
    """Per-IP sliding window on login attempts (§8).

    In-memory on purpose: a BOP runs one API process, and a Redis dependency for
    rate limiting would make login fail when Redis does — exactly backwards.
    """

    def __init__(self, per_minute: int = 5) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, ip: str) -> None:
        now = time.monotonic()
        window = self._hits[ip]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.per_minute:
            logger.warning("login rate limit hit from ip=%s", ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="too many login attempts; try again in a minute",
            )
        window.append(now)

    def reset(self, ip: str) -> None:
        self._hits.pop(ip, None)


login_limiter = LoginRateLimiter()


class Principal:
    """The authenticated caller."""

    __slots__ = ("role", "token_type", "user_id")

    def __init__(self, user_id: str, role: str, token_type: str = "access") -> None:
        self.user_id = user_id
        self.role = role
        self.token_type = token_type

    def at_least(self, required: str) -> bool:
        try:
            return ROLE_ORDER.index(self.role) >= ROLE_ORDER.index(required)
        except ValueError:
            return False

    def __repr__(self) -> str:  # pragma: no cover
        return f"Principal(user_id={self.user_id!r}, role={self.role!r})"


async def current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> Principal:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials, settings)
    if payload.get("typ") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh tokens cannot be used to call the API",
        )
    principal = Principal(payload["sub"], payload.get("role", "viewer"))
    request.state.principal = principal
    return principal


def require_role(minimum: str):
    """Dependency factory. RBAC lives here, not in router bodies (§8)."""
    if minimum not in ROLE_ORDER:
        raise ValueError(f"unknown role {minimum!r}; expected one of {ROLE_ORDER}")

    async def _dependency(
        principal: Annotated[Principal, Depends(current_principal)],
    ) -> Principal:
        if not principal.at_least(minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"this action requires the {minimum} role or higher; "
                f"you have {principal.role}",
            )
        return principal

    return _dependency


RequireViewer = Annotated[Principal, Depends(require_role("viewer"))]
RequireOperator = Annotated[Principal, Depends(require_role("operator"))]
RequireInvestigator = Annotated[Principal, Depends(require_role("investigator"))]
RequireAdmin = Annotated[Principal, Depends(require_role("admin"))]
