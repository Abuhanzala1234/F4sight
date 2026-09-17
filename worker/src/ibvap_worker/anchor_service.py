"""Ledger anchoring service (BUILD_SPEC §7.12, §3.3 step 12).

Runs as its own process (`make anchor`). Every ``anchor_interval_s`` it:

1. claims the alerts whose ``ledger_status='pending'``;
2. builds a Merkle tree over their ``evidence_hash`` values;
3. writes the root to the ledger;
4. stores each alert's inclusion proof and flips it to ``anchored``.

The invariant this service exists to protect: **ledger unavailability never
blocks alerting**. This process can be down for a week and not one alert is
lost or delayed — they simply stay ``pending`` until it comes back. That is why
anchoring is a separate process rather than a step in the pipeline.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Any

from .config import load_config
from .ledger import build_ledger
from .logsetup import configure_logging
from .merkle import build_tree, proof

logger = logging.getLogger(__name__)

__all__ = ["AnchorService", "main"]


class AnchorService:
    def __init__(self, dsn: str, ledger: Any, interval_s: float = 30.0, max_leaves: int = 512):
        self.dsn = dsn
        self.ledger = ledger
        self.interval_s = interval_s
        self.max_leaves = max_leaves
        self._stop = False
        self._pool: Any = None

    def _connection(self) -> Any:
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(self.dsn, min_size=1, max_size=2, open=True)
        return self._pool.connection()

    def run(self) -> None:
        logger.info(
            "anchor service started backend=%s interval=%.0fs max_leaves=%d",
            getattr(self.ledger, "backend", "?"),
            self.interval_s,
            self.max_leaves,
        )
        while not self._stop:
            try:
                anchored = self.tick()
                if anchored:
                    logger.info("anchored %d alerts", anchored)
            except Exception:
                # Retry forever (config: retry_forever). A ledger outage is an
                # expected state, not a reason to exit.
                logger.exception("anchor tick failed; will retry next interval")
            for _ in range(int(self.interval_s * 10)):
                if self._stop:
                    break
                time.sleep(0.1)
        logger.info("anchor service stopped")

    def stop(self) -> None:
        self._stop = True

    def tick(self) -> int:
        """One batch. Returns the number of alerts anchored."""
        from psycopg.types.json import Jsonb

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, evidence_hash, site_id
                FROM alert
                WHERE ledger_status = 'pending' AND evidence_hash IS NOT NULL
                ORDER BY ts_utc
                LIMIT %s
                """,
                (self.max_leaves,),
            ).fetchall()

            if not rows:
                return 0

            alert_ids = [r[0] for r in rows]
            leaves = [r[1] for r in rows]
            tree = build_tree(leaves)

            health = self.ledger.health()
            if not health.available:
                logger.warning(
                    "ledger unavailable (%s); %d alerts stay pending. " "Alerting is unaffected.",
                    health.detail,
                    len(rows),
                )
                return 0

            receipt = self.ledger.anchor(
                tree.root,
                {
                    "leaf_count": tree.leaf_count,
                    "alert_ids": [str(a) for a in alert_ids],
                },
            )

            batch_row = conn.execute(
                """
                INSERT INTO ledger_anchor_batch
                    (id, merkle_root, leaf_count, backend, tx_id, block_number,
                     anchored_at, status)
                VALUES (uuid7(), %s, %s, %s, %s, %s, %s, 'anchored')
                RETURNING id
                """,
                (
                    tree.root,
                    tree.leaf_count,
                    receipt.backend,
                    receipt.tx_id,
                    receipt.block_number,
                    receipt.anchored_at,
                ),
            ).fetchone()
            batch_id = batch_row[0]

            for index, (alert_id, leaf) in enumerate(zip(alert_ids, leaves, strict=True)):
                path = proof(tree, index)
                conn.execute(
                    """
                    INSERT INTO ledger_anchor_entry
                        (id, batch_id, alert_id, leaf_hash, leaf_index, proof)
                    VALUES (uuid7(), %s, %s, %s, %s, %s)
                    """,
                    (batch_id, alert_id, leaf, index, Jsonb([list(p) for p in path])),
                )
                conn.execute(
                    "UPDATE alert SET ledger_status = 'anchored' WHERE id = %s",
                    (alert_id,),
                )

            conn.commit()
            return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IBVAP ledger anchoring service")
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--site", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, profile=args.profile, site=args.site)
    configure_logging(cfg.get("logging.level", "INFO"), cfg.get("logging.format", "console"))

    import os

    dsn = (
        f"postgresql://{os.getenv('DB_USER', 'ibvap')}:"
        f"{os.getenv('DB_PASSWORD', 'ibvap_dev')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'ibvap')}"
    )

    service = AnchorService(
        dsn=dsn,
        ledger=build_ledger(cfg.as_dict()),
        interval_s=float(cfg.get("ledger.anchor_interval_s", 30)),
        max_leaves=int(cfg.get("ledger.max_batch_leaves", 512)),
    )

    signal.signal(signal.SIGINT, lambda *_: service.stop())
    signal.signal(signal.SIGTERM, lambda *_: service.stop())
    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
