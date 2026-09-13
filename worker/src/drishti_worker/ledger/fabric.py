"""Hyperledger Fabric ledger backend (BUILD_SPEC §7.12). **[STRETCH backend]**

Fabric is the theme of the problem statement and it is also, per CLAUDE.md, the
highest-variance task in the project. This class exists so that turning it on is
``ledger.backend: fabric`` in config and nothing else.

It talks to the peer through the Fabric Gateway (``fabric-gateway`` gRPC, free,
Apache-2.0), which is the supported path for Fabric 2.5 and removes the old SDK's
dependency chain.

Everything here is best-effort by design. Every failure path returns or raises
in a way the anchor service already handles by re-queuing, because the invariant
is absolute: **ledger unavailability never blocks alerting**.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import AnchorReceipt, AnchorRecord, LedgerHealth

logger = logging.getLogger(__name__)

__all__ = ["FabricLedger"]


class FabricLedger:
    """Anchors Merkle roots to the ``evidencecc`` chaincode on channel ``drishti``."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.channel = str(cfg.get("channel", "drishti"))
        self.chaincode = str(cfg.get("chaincode", "evidencecc"))
        self.msp_id = str(cfg.get("msp_id", "Org1MSP"))
        self.peer_endpoint = str(cfg.get("peer_endpoint", "localhost:7051"))
        self.timeout_s = float(cfg.get("gateway_timeout_s", 10))
        self.crypto_path = Path(cfg.get("crypto_path", "fabric/organizations"))
        self._gateway: Any = None
        self._contract: Any = None

    @property
    def backend(self) -> str:
        return "fabric"

    def _connect(self) -> Any:
        if self._contract is not None:
            return self._contract
        try:
            import grpc
            from hfc.gateway import Gateway, Identity, Signer
        except ImportError as exc:
            raise RuntimeError(
                "Fabric backend selected but the gateway SDK is not installed. "
                "Install `fabric-gateway` and `grpcio`, or set ledger.backend=mock "
                "in config/ledger.yaml (the default, and what `make demo` uses)."
            ) from exc

        cert_path = self.crypto_path / "users" / "User1" / "msp" / "signcerts" / "cert.pem"
        key_dir = self.crypto_path / "users" / "User1" / "msp" / "keystore"
        tls_path = self.crypto_path / "peers" / "peer0" / "tls" / "ca.crt"
        for path in (cert_path, tls_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Fabric crypto material missing: {path}. Run `make fabric-up` first."
                )

        credentials = grpc.ssl_channel_credentials(tls_path.read_bytes())
        channel = grpc.secure_channel(self.peer_endpoint, credentials)
        identity = Identity(self.msp_id, cert_path.read_bytes())
        signer = Signer(next(key_dir.glob("*")).read_bytes())

        self._gateway = Gateway.connect(client=channel, identity=identity, signer=signer)
        network = self._gateway.get_network(self.channel)
        self._contract = network.get_contract(self.chaincode)
        logger.info(
            "fabric gateway connected peer=%s channel=%s chaincode=%s",
            self.peer_endpoint,
            self.channel,
            self.chaincode,
        )
        return self._contract

    def anchor(self, merkle_root: str, meta: Mapping[str, Any]) -> AnchorReceipt:
        contract = self._connect()
        anchored_at = datetime.now(UTC).isoformat()
        result = contract.submit_transaction(
            "AnchorBatch",
            merkle_root,
            str(meta.get("leaf_count", 0)),
            str(meta.get("site_code", "")),
            anchored_at,
        )
        tx_id = getattr(result, "transaction_id", None) or str(result)
        logger.info("anchored root=%s… tx=%s backend=fabric", merkle_root[:16], tx_id)
        return AnchorReceipt(
            tx_id=tx_id,
            merkle_root=merkle_root,
            anchored_at=anchored_at,
            backend="fabric",
            block_number=getattr(result, "block_number", None),
            meta=dict(meta),
        )

    def get(self, tx_id: str) -> AnchorRecord | None:
        try:
            contract = self._connect()
            payload = contract.evaluate_transaction("GetAnchor", tx_id)
        except Exception:
            logger.exception("fabric GetAnchor failed for tx=%s", tx_id)
            return None
        if not payload:
            return None
        import json

        data = json.loads(payload)
        return AnchorRecord(
            tx_id=data["tx_id"],
            merkle_root=data["merkle_root"],
            anchored_at=data["anchored_at"],
            backend="fabric",
            block_number=data.get("block_number"),
            leaf_count=int(data.get("leaf_count", 0)),
            meta=data.get("meta", {}),
        )

    def health(self) -> LedgerHealth:
        started = time.monotonic()
        try:
            self._connect()
            return LedgerHealth(
                available=True,
                backend="fabric",
                detail=f"peer={self.peer_endpoint} channel={self.channel}",
                latency_ms=round((time.monotonic() - started) * 1000, 1),
            )
        except Exception as exc:
            # Not an error-level event: an unreachable ledger is an expected
            # state at a BOP (P9), and alerts keep firing regardless.
            logger.warning("fabric unavailable (%s); anchors will queue", exc)
            return LedgerHealth(available=False, backend="fabric", detail=str(exc))
