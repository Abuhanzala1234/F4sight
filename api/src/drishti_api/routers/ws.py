"""WebSocket alert fan-out (BUILD_SPEC §9).

Reads the Redis Stream the worker publishes to and pushes to every subscribed
dashboard. One reader task per API process, not one per client — a hundred
open dashboards must not mean a hundred Redis consumers.

The contract that matters: **a dropped socket must never mean a lost alert on
screen.** The client reconnects with backoff and replays via
``GET /alerts?from=<last_seen_ts>``. This socket is an accelerator for the REST
feed, never the only way to learn something happened.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..security import ROLE_ORDER, decode_token
from ..settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["realtime"])

SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"]


class Subscription:
    __slots__ = ("min_severity", "site_ids", "socket", "user_id")

    def __init__(self, socket: WebSocket, user_id: str) -> None:
        self.socket = socket
        self.user_id = user_id
        self.site_ids: set[str] = set()
        self.min_severity = "info"

    def wants(self, alert: dict[str, Any]) -> bool:
        if self.site_ids and alert.get("site_id") not in self.site_ids:
            return False
        try:
            return SEVERITY_ORDER.index(
                alert.get("severity", "info")
            ) >= SEVERITY_ORDER.index(self.min_severity)
        except ValueError:
            return True  # unknown severity: show it rather than hide it


class ConnectionManager:
    """Owns the client set and the single Redis reader task."""

    def __init__(self) -> None:
        self._subs: set[Subscription] = set()
        self._reader: asyncio.Task[None] | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def connect(self, sub: Subscription) -> None:
        async with self._lock:
            self._subs.add(sub)
            if self._reader is None or self._reader.done():
                self._reader = asyncio.create_task(self._pump(), name="ws-redis-pump")
            if self._heartbeat is None or self._heartbeat.done():
                self._heartbeat = asyncio.create_task(self._beat(), name="ws-heartbeat")

    async def disconnect(self, sub: Subscription) -> None:
        async with self._lock:
            self._subs.discard(sub)

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[Subscription] = []
        for sub in list(self._subs):
            if message.get("type") == "alert" and not sub.wants(
                message.get("alert", {})
            ):
                continue
            try:
                await sub.socket.send_json(message)
            except Exception:
                dead.append(sub)
        for sub in dead:
            await self.disconnect(sub)

    async def _pump(self) -> None:
        """Tail the Redis Stream the worker writes to (§7.13)."""
        settings = get_settings()
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.error("redis package missing; live alert push is disabled")
            return

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        last_id = "$"  # only new entries; history comes from REST
        logger.info("websocket pump tailing %s", settings.alert_stream)

        while self._subs:
            try:
                entries = await client.xread(
                    {settings.alert_stream: last_id}, count=32, block=2000
                )
            except Exception:
                # Redis down degrades fan-out to polling; the DB path is
                # unaffected (§13). Retry rather than kill the task.
                logger.exception("redis xread failed; retrying in 5s")
                await asyncio.sleep(5)
                continue

            for _stream, messages in entries or []:
                for message_id, fields in messages:
                    last_id = message_id
                    try:
                        alert = json.loads(fields["payload"])
                    except (KeyError, json.JSONDecodeError):
                        logger.exception("malformed alert on the stream: %r", fields)
                        continue
                    await self.broadcast({"type": "alert", "alert": alert})

        with contextlib.suppress(Exception):
            await client.aclose()
        logger.info("websocket pump stopped (no subscribers)")

    async def _beat(self) -> None:
        while self._subs:
            await asyncio.sleep(15)
            await self.broadcast(
                {"type": "heartbeat", "ts": datetime.now(UTC).isoformat()}
            )


manager = ConnectionManager()


@router.websocket("/ws/alerts")
async def alerts_socket(socket: WebSocket, token: str = Query(...)) -> None:
    """Live alert feed.

    The token comes as a query parameter because browsers cannot set headers on
    a WebSocket handshake. It is still a real JWT and still checked; it is
    logged nowhere.
    """
    settings = get_settings()
    try:
        payload = decode_token(token, settings)
    except Exception:
        await socket.close(code=4401, reason="invalid token")
        return
    if payload.get("role") not in ROLE_ORDER:
        await socket.close(code=4403, reason="insufficient role")
        return

    await socket.accept()
    sub = Subscription(socket, payload["sub"])
    await manager.connect(sub)
    await socket.send_json(
        {
            "type": "hello",
            "server_time": datetime.now(UTC).isoformat(),
            "subscribed": [],
            "note": "on reconnect, replay with GET /api/v1/alerts?from=<last_seen_ts>",
        }
    )

    try:
        while True:
            message = await socket.receive_json()
            if message.get("type") == "subscribe":
                sub.site_ids = set(message.get("site_ids") or [])
                sub.min_severity = message.get("min_severity", "info")
                await socket.send_json(
                    {
                        "type": "hello",
                        "server_time": datetime.now(UTC).isoformat(),
                        "subscribed": sorted(sub.site_ids),
                    }
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("websocket error for user=%s", sub.user_id)
    finally:
        await manager.disconnect(sub)
