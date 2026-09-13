#!/usr/bin/env python3
"""Replay spooled alerts into Postgres after an outage (BUILD_SPEC §13).

When the database is unreachable the worker writes alerts to a local JSONL
spool rather than dropping them — *never lose an alert*. This replays that
spool once the database is back, and only deletes a file after every line in it
has been committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level="INFO", format="%(levelname)-5s %(message)s")
logger = logging.getLogger("replay")


def dsn() -> str:
    return (
        f"postgresql://{os.getenv('DB_USER', 'drishti')}:"
        f"{os.getenv('DB_PASSWORD', 'drishti_dev')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'drishti')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", default="spool")
    parser.add_argument(
        "--delete", action="store_true", help="remove files once fully replayed"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spool = Path(args.spool)
    files = sorted(spool.glob("alerts-*.jsonl"))
    if not files:
        logger.info("nothing to replay in %s", spool)
        return 0

    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError:
        logger.error("psycopg is required: pip install 'psycopg[binary]'")
        return 1

    total = replayed = failed = 0
    with psycopg.connect(dsn()) as conn:
        for path in files:
            file_ok = True
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                total += 1
                try:
                    alert = json.loads(line)
                except json.JSONDecodeError:
                    logger.exception("%s:%d is not valid JSON; skipping", path, line_no)
                    failed += 1
                    file_ok = False
                    continue

                if args.dry_run:
                    logger.info(
                        "would replay %s (%s)", alert["alert_id"], alert["kind"]
                    )
                    continue

                try:
                    with conn.transaction():
                        # ON CONFLICT DO NOTHING makes replay idempotent, so
                        # running this twice is harmless.
                        conn.execute(
                            """
                            INSERT INTO alert (
                                id, site_id, camera_id, track_id, zone_id, kind,
                                severity, risk_score, risk_breakdown, reason_codes,
                                ts_utc, window_start, window_end, status,
                                evidence_hash, evidence_doc, ledger_status
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (
                                alert["alert_id"],
                                alert["site_id"],
                                alert["camera_id"],
                                alert["track_id"],
                                alert["zone_id"],
                                alert["kind"],
                                alert["severity"],
                                alert["risk_score"],
                                Jsonb(alert["risk_breakdown"]),
                                alert["reason_codes"],
                                alert["ts_utc"],
                                alert["window_start"],
                                alert["window_end"],
                                alert["status"],
                                alert["evidence_hash"],
                                Jsonb(alert["evidence_doc"]),
                                "pending",
                            ),
                        )
                    replayed += 1
                except Exception:
                    logger.exception("failed to replay %s", alert.get("alert_id"))
                    failed += 1
                    file_ok = False

            # Only delete once every line landed. A half-replayed file that
            # gets deleted is exactly the alert loss the spool exists to prevent.
            if args.delete and file_ok and not args.dry_run:
                path.unlink()
                logger.info("replayed and removed %s", path)

    logger.info("replayed %d/%d alerts (%d failed)", replayed, total, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
