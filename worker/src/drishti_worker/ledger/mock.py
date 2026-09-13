"""Mock ledger: an append-only, hash-chained JSONL file (BUILD_SPEC §7.12).

This is not a toy stand-in that returns success. It is a real append-only log
where each entry carries the hash of the previous one, so the file itself is
tamper-evident: altering entry 4 invalidates the chain for 5 onward, and
``verify_chain()`` says exactly where.

That property is what makes the mock a legitimate demo backend. It exercises the
same code path, produces the same receipts, and supports the same verification
story — it simply has one writer instead of an endorsement policy. When Fabric
comes up, only the class changes.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from . import AnchorReceipt, AnchorRecord, LedgerHealth

logger = logging.getLogger(__name__)

__all__ = ["MockLedger"]

GENESIS = "0" * 64


class MockLedger:
    def __init__(self, path: str | Path = "ledger_data/mock_ledger.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._height = self._count_entries()

    @property
    def backend(self) -> str:
        return "mock"

    def anchor(self, merkle_root: str, meta: Mapping[str, Any]) -> AnchorReceipt:
        if len(merkle_root) != 64:
            raise ValueError(
                f"merkle root must be a 64-char hex digest, got {len(merkle_root)}"
            )

        with self._lock:
            prev_hash = self._tail_hash()
            block_number = self._height
            anchored_at = datetime.now(UTC).isoformat()
            entry: dict[str, Any] = {
                "block_number": block_number,
                "merkle_root": merkle_root,
                "anchored_at": anchored_at,
                "prev_hash": prev_hash,
                "meta": dict(meta),
            }
            # The chain link covers everything above, so an edit anywhere breaks
            # every subsequent link.
            entry_hash = sha256(
                json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            entry["entry_hash"] = entry_hash
            entry["tx_id"] = f"mock:{block_number}:{entry_hash[:16]}"

            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
                )
                fh.flush()
            self._height += 1

        logger.info(
            "anchored root=%s… block=%d leaves=%s backend=mock",
            merkle_root[:16],
            block_number,
            meta.get("leaf_count"),
        )
        return AnchorReceipt(
            tx_id=entry["tx_id"],
            merkle_root=merkle_root,
            anchored_at=anchored_at,
            backend="mock",
            block_number=block_number,
            meta=dict(meta),
        )

    def get(self, tx_id: str) -> AnchorRecord | None:
        for entry in self._iter_entries():
            if entry.get("tx_id") == tx_id:
                return AnchorRecord(
                    tx_id=entry["tx_id"],
                    merkle_root=entry["merkle_root"],
                    anchored_at=entry["anchored_at"],
                    backend="mock",
                    block_number=entry.get("block_number"),
                    leaf_count=int(entry.get("meta", {}).get("leaf_count", 0)),
                    meta=entry.get("meta", {}),
                )
        return None

    def get_by_root(self, merkle_root: str) -> AnchorRecord | None:
        for entry in self._iter_entries():
            if entry.get("merkle_root") == merkle_root:
                return self.get(entry["tx_id"])
        return None

    def health(self) -> LedgerHealth:
        try:
            writable = self.path.parent.exists()
            return LedgerHealth(
                available=writable,
                backend="mock",
                detail=f"height={self._height} path={self.path}",
            )
        except OSError as exc:
            logger.exception("mock ledger health check failed")
            return LedgerHealth(available=False, backend="mock", detail=str(exc))

    def verify_chain(self) -> tuple[bool, str]:
        """Walk the chain and report the first break.

        Returns ``(ok, detail)``. Loud, specific failure — P5.
        """
        prev = GENESIS
        for index, entry in enumerate(self._iter_entries()):
            if entry.get("prev_hash") != prev:
                return False, (
                    f"chain broken at block {index}: prev_hash="
                    f"{entry.get('prev_hash', '')[:16]}… expected {prev[:16]}…"
                )
            stored = entry.get("entry_hash", "")
            recomputed = sha256(
                json.dumps(
                    {
                        k: v
                        for k, v in entry.items()
                        if k not in ("entry_hash", "tx_id")
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if stored != recomputed:
                return False, (
                    f"entry {index} was modified: stored hash {stored[:16]}… "
                    f"but content hashes to {recomputed[:16]}…"
                )
            prev = stored
        return True, f"chain intact across {self._height} entries"

    # -- internals ---------------------------------------------------------

    def _iter_entries(self) -> Any:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.exception(
                        "mock ledger line %d is not valid JSON; skipping it", line_no
                    )

    def _count_entries(self) -> int:
        return sum(1 for _ in self._iter_entries())

    def _tail_hash(self) -> str:
        last = GENESIS
        for entry in self._iter_entries():
            last = entry.get("entry_hash", last)
        return last
