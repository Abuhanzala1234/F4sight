"""Ledger backends (§7.12). The mock is a real hash-chained log, not a stub."""

from __future__ import annotations

import hashlib
import json

import pytest

from ibvap_worker.ledger import build_ledger
from ibvap_worker.ledger.mock import MockLedger


def root(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture
def ledger(tmp_path):
    return MockLedger(tmp_path / "ledger.jsonl")


class TestMockLedger:
    def test_anchor_returns_a_receipt(self, ledger):
        receipt = ledger.anchor(root("a"), {"leaf_count": 3, "site_code": "BOP-03"})
        assert receipt.backend == "mock"
        assert receipt.merkle_root == root("a")
        assert receipt.block_number == 0
        assert receipt.tx_id.startswith("mock:")

    def test_block_numbers_increment(self, ledger):
        blocks = [ledger.anchor(root(str(i)), {"leaf_count": 1}).block_number for i in range(4)]
        assert blocks == [0, 1, 2, 3]

    def test_get_round_trip(self, ledger):
        receipt = ledger.anchor(root("a"), {"leaf_count": 5})
        record = ledger.get(receipt.tx_id)
        assert record.merkle_root == root("a")
        assert record.leaf_count == 5

    def test_get_by_root(self, ledger):
        ledger.anchor(root("z"), {"leaf_count": 1})
        assert ledger.get_by_root(root("z")) is not None

    def test_get_unknown_returns_none(self, ledger):
        assert ledger.get("mock:999:deadbeef") is None

    def test_bad_root_refused(self, ledger):
        with pytest.raises(ValueError):
            ledger.anchor("too-short", {})

    def test_health(self, ledger):
        assert ledger.health().available

    def test_survives_a_restart(self, tmp_path):
        path = tmp_path / "l.jsonl"
        MockLedger(path).anchor(root("a"), {"leaf_count": 1})
        second = MockLedger(path)
        receipt = second.anchor(root("b"), {"leaf_count": 1})
        assert receipt.block_number == 1


class TestTamperEvidence:
    """The mock is a legitimate demo backend precisely because of this."""

    def test_intact_chain_verifies(self, ledger):
        for i in range(5):
            ledger.anchor(root(str(i)), {"leaf_count": i})
        ok, detail = ledger.verify_chain()
        assert ok, detail

    def test_modified_entry_is_detected_with_a_location(self, tmp_path):
        path = tmp_path / "l.jsonl"
        led = MockLedger(path)
        for i in range(5):
            led.anchor(root(str(i)), {"leaf_count": i})

        lines = path.read_text().splitlines()
        entry = json.loads(lines[2])
        entry["meta"]["leaf_count"] = 9999
        lines[2] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n")

        ok, detail = MockLedger(path).verify_chain()
        assert not ok
        assert "2" in detail  # says exactly where

    def test_deleted_entry_breaks_the_chain(self, tmp_path):
        path = tmp_path / "l.jsonl"
        led = MockLedger(path)
        for i in range(5):
            led.anchor(root(str(i)), {"leaf_count": i})
        lines = path.read_text().splitlines()
        del lines[2]
        path.write_text("\n".join(lines) + "\n")

        ok, detail = MockLedger(path).verify_chain()
        assert not ok
        assert "chain broken" in detail


class TestFactory:
    def test_mock_is_the_default(self, tmp_path):
        led = build_ledger({"ledger": {"mock": {"path": str(tmp_path / "l.jsonl")}}})
        assert led.health().backend == "mock"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown ledger backend"):
            build_ledger({"ledger": {"backend": "bitcoin"}})
