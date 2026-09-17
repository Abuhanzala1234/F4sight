"""The worker's read-only view of the watchlist tables (§7.9, §7.10).

Deliberately separate from ``anpr.py`` and ``faces.py``: those are pure,
property-tested matching logic with no native dependencies; this is the one
place in either path that talks to Postgres, and it does so fail-soft. A
watchlist that cannot be refreshed just means no *new* hits are recognised
until the next refresh succeeds — never a worker crash, and never a stale
match either, since a failed refresh keeps the last known-good table instead
of clearing it (P9: offline is normal, not an error).

The plate cache stores nothing but what ``WatchlistVehicle`` already holds
(``plate_hmac``, ``category``) — no plaintext plate ever passes through here,
matching P6 the same way ``anpr.py`` does. The face cache is the one place in
the worker that ever loads an enrolled embedding, and it only ever *reads*
one — enrolment itself is API-only and audited (faces.py's module docstring;
this file has no write path to either watchlist table).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from .faces import FaceMatch, FaceWatchlist, WatchlistFace

logger = logging.getLogger(__name__)

__all__ = ["FaceWatchlistCache", "PlateWatchHit", "WatchlistCache"]


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


def _parse_pgvector(raw: str) -> tuple[float, ...]:
    """pgvector's text form is ``[0.1,0.2,...]``. Parsed by hand rather than
    via the ``pgvector`` client package so the worker does not need it just to
    read a watchlist that most deployments never turn on (P6, P9)."""
    return tuple(float(x) for x in raw.strip("[]").split(","))


class FaceWatchlistCache:
    """Thread-safe wrapper around ``faces.FaceWatchlist`` (§7.10).

    Same shape as ``WatchlistCache`` above and for the same reason: the
    refresh thread swaps the whole table atomically under a lock, the stage
    thread's ``match`` calls never see a half-loaded watchlist.
    """

    def __init__(self, threshold: float = 0.55) -> None:
        self._lock = threading.Lock()
        self._inner = FaceWatchlist(threshold=threshold)

    def match(self, embedding: tuple[float, ...]) -> FaceMatch | None:
        with self._lock:
            return self._inner.match(embedding)

    @property
    def count(self) -> int:
        with self._lock:
            return self._inner.count

    def set_faces(self, faces: list[WatchlistFace]) -> None:
        with self._lock:
            self._inner.set_faces(faces)

    def refresh_from_db(self, dsn: str) -> bool:
        """One best-effort reload. Returns whether it succeeded.

        Joins through ``watchlist_person`` so a deactivated or expired person
        drops out of matching immediately, even though their embedding rows
        are untouched — the same "expired entries must not still match" rule
        ``WatchlistCache.refresh_from_db`` applies to plates (P2).
        """
        try:
            import psycopg

            with psycopg.connect(dsn, connect_timeout=3) as conn:
                rows = conn.execute("""
                    SELECT p.id, p.ref_code, p.category, v.embedding
                    FROM watchlist_face_vector v
                    JOIN watchlist_person p ON p.id = v.person_id
                    WHERE p.active AND (p.expires_at IS NULL OR p.expires_at > now())
                    """).fetchall()
        except Exception as exc:
            logger.warning(
                "face watchlist refresh failed (%s); keeping the last known %d entries",
                exc,
                self.count,
            )
            return False

        faces = [
            WatchlistFace(
                person_id=str(r[0]),
                ref_code=r[1],
                category=r[2],
                embedding=_parse_pgvector(r[3]) if isinstance(r[3], str) else tuple(r[3]),
            )
            for r in rows
        ]
        self.set_faces(faces)
        logger.info("face watchlist refreshed: %d active entries", len(faces))
        return True

    def start_refresh_thread(self, dsn: str, interval_s: float = 60.0) -> threading.Thread:
        stop = threading.Event()

        def _loop() -> None:
            while not stop.is_set():
                self.refresh_from_db(dsn)
                stop.wait(interval_s)

        thread = threading.Thread(target=_loop, name="face-watchlist-refresh", daemon=True)
        thread.start()
        return thread
