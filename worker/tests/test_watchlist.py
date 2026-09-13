"""Watchlist cache (§7.9): thread-safe, fail-soft, no plaintext ever touches
it (P6). ``refresh_from_db`` is exercised against a bad DSN only -- a real
Postgres is exactly what CLAUDE.md says to mock here, not the logic.
"""

from __future__ import annotations

import threading

from drishti_worker.watchlist import PlateWatchHit, WatchlistCache


class TestPlateMatch:
    def test_no_match_returns_none(self):
        cache = WatchlistCache()
        assert cache.plate_match("a" * 64) is None

    def test_injected_entry_matches_by_hmac(self):
        cache = WatchlistCache()
        cache.set_plates([PlateWatchHit("b" * 64, "stolen")])
        hit = cache.plate_match("b" * 64)
        assert hit is not None
        assert hit.category == "stolen"

    def test_set_plates_replaces_the_whole_table(self):
        cache = WatchlistCache()
        cache.set_plates([PlateWatchHit("a" * 64, "watch")])
        cache.set_plates([PlateWatchHit("b" * 64, "watch")])
        assert cache.plate_match("a" * 64) is None
        assert cache.plate_match("b" * 64) is not None

    def test_plate_count_reflects_the_current_table(self):
        cache = WatchlistCache()
        assert cache.plate_count == 0
        cache.set_plates([PlateWatchHit("a" * 64), PlateWatchHit("b" * 64)])
        assert cache.plate_count == 2


class TestRefreshIsFailSoft:
    def test_unreachable_database_does_not_raise(self):
        cache = WatchlistCache()
        # No psycopg server is listening on this port; connect_timeout=3 means
        # this could be slow, so we assert on the contract, not the timing.
        ok = cache.refresh_from_db("postgresql://nobody:nothing@127.0.0.1:1/nope")
        assert ok is False

    def test_a_failed_refresh_keeps_the_previous_table(self):
        cache = WatchlistCache()
        cache.set_plates([PlateWatchHit("a" * 64, "watch")])
        cache.refresh_from_db("postgresql://nobody:nothing@127.0.0.1:1/nope")
        # P9: offline is normal. A refresh failure must not clear known hits.
        assert cache.plate_match("a" * 64) is not None


class TestThreadSafety:
    def test_concurrent_reads_during_a_swap_never_raise(self):
        cache = WatchlistCache()
        cache.set_plates([PlateWatchHit(f"{i:064d}") for i in range(50)])
        stop = threading.Event()
        errors: list[Exception] = []

        def reader() -> None:
            while not stop.is_set():
                try:
                    cache.plate_match(
                        "00000000000000000000000000000000000000000000000000000000000001"
                    )
                except Exception as exc:  # pragma: no cover - failure path only
                    errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for i in range(20):
            cache.set_plates([PlateWatchHit(f"{i:064d}")])
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        assert errors == []
