"""The worker's read-only view of the watchlist tables (§7.9, §7.10).

Deliberately separate from ``anpr.py``: that module is pure, property-tested
matching logic with no native dependencies; this is the one place in the
plate-matching path that talks to Postgres, and it does so fail-soft. A
watchlist that cannot be refreshed just means no *new* hits are recognised
until the next refresh succeeds — never a worker crash, and never a stale
match either, since a failed refresh keeps the last known-good table instead
of clearing it (P9: offline is normal, not an error).

The cache stores nothing but what ``WatchlistVehicle`` already holds
(``plate_hmac``, ``category``) — no plaintext plate ever passes through here,
matching P6 the same way ``anpr.py`` does.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["PlateWatchHit", "WatchlistCache"]


@dataclass(frozen=True, slots=True)
class PlateWatchHit:
    plate_hmac: str
    category: str = "watch"


class WatchlistCache:
    """Thread-safe snapshot, swapped atomically on refresh so a reader never
    sees a half-updated table (the stage thread calls ``plate_match`` on every
    settled plate; the refresh thread replaces ``_plates`` wholesale under the
    lock, never mutates it in place).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plates: dict[str, PlateWatchHit] = {}

    def plate_match(self, plate_hmac: str) -> PlateWatchHit | None:
        with self._lock:
            return self._plates.get(plate_hmac)

    def set_plates(self, entries: list[PlateWatchHit]) -> None:
        """Direct injection: what tests use, and what ``refresh_from_db`` uses
        internally once it has a result to install."""
        with self._lock:
            self._plates = {e.plate_hmac: e for e in entries}

    @property
    def plate_count(self) -> int:
        with self._lock:
            return len(self._plates)

    def refresh_from_db(self, dsn: str) -> bool:
        """One best-effort reload. Returns whether it succeeded.

        Only active, unexpired entries are loaded — an expired watchlist entry
        that still matched would be a false positive nobody could explain,
        which is exactly what P2 rules out.
        """
        try:
            import psycopg

            with psycopg.connect(dsn, connect_timeout=3) as conn:
                rows = conn.execute("""
                    SELECT plate_hmac, category FROM watchlist_vehicle
                    WHERE active AND (expires_at IS NULL OR expires_at > now())
                    """).fetchall()
        except Exception as exc:
            logger.warning(
                "watchlist refresh failed (%s); keeping the last known %d " "plate entries",
                exc,
                self.plate_count,
            )
            return False

        self.set_plates([PlateWatchHit(r[0], r[1]) for r in rows])
        logger.info("watchlist refreshed: %d active vehicle entries", len(rows))
        return True

    def start_refresh_thread(self, dsn: str, interval_s: float = 60.0) -> threading.Thread:
        """Background poller, daemon so it never blocks shutdown. Refreshes
        once immediately so the first alerts after startup already see today's
        watchlist rather than waiting a full interval."""

        stop = threading.Event()

        def _loop() -> None:
            while not stop.is_set():
                self.refresh_from_db(dsn)
                stop.wait(interval_s)

        thread = threading.Thread(target=_loop, name="watchlist-refresh", daemon=True)
        thread.start()
        return thread
