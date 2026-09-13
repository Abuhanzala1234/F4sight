"""Ledger backends for evidence anchoring (BUILD_SPEC §7.12).

Blocker #6, quoting CLAUDE.md:

    Fabric setup. Highest-variance task in the project. Build the mock ledger
    backend first and wire everything to the interface. Fabric never blocks the
    critical path.

So: one protocol, two implementations, and ``backend: mock`` is the default in
``config/ledger.yaml``. The demo runs on the mock. Fabric is a config change.

And the invariant that outranks both: **ledger unavailability never blocks
alerting**. An alert with ``ledger_status='pending'`` is a perfectly good alert.
Queue the anchor, fire the alert.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AnchorReceipt",
    "AnchorRecord",
    "LedgerBackend",
    "LedgerHealth",
    "build_ledger",
]


@dataclass(frozen=True, slots=True)
class AnchorReceipt:
    tx_id: str
    merkle_root: str
    anchored_at: str  # ISO-8601 UTC
    backend: str
    block_number: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    tx_id: str
    merkle_root: str
    anchored_at: str
    backend: str
    block_number: int | None = None
    leaf_count: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LedgerHealth:
    available: bool
    backend: str
    detail: str = ""
    latency_ms: float | None = None


@runtime_checkable
class LedgerBackend(Protocol):
    """Frozen interface (§7.12)."""

    def anchor(self, merkle_root: str, meta: Mapping[str, Any]) -> AnchorReceipt: ...

    def get(self, tx_id: str) -> AnchorRecord | None: ...

    def health(self) -> LedgerHealth: ...


def build_ledger(cfg: Mapping[str, Any]) -> LedgerBackend:
    """Factory. ``mock`` is the default and always works offline (P9)."""
    block = dict(cfg.get("ledger", cfg))
    backend = str(block.get("backend", "mock")).lower()

    if backend == "mock":
        from .mock import MockLedger

        return MockLedger(
            block.get("mock", {}).get("path", "ledger_data/mock_ledger.jsonl")
        )
    if backend == "fabric":
        from .fabric import FabricLedger

        return FabricLedger(block.get("fabric", {}))
    raise ValueError(f"unknown ledger backend {backend!r}; expected 'mock' or 'fabric'")
