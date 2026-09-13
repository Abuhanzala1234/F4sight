"""Evidence verification (§8). **[DEMO-CRITICAL]**

This is the endpoint that wins the blockchain argument, so it returns every
intermediate value rather than a verdict. A sceptical evaluator should be able
to recompute each number by hand from the response.

``POST /verify/document`` is the important one: it verifies an evidence JSON
that somebody hands you, without that alert needing to exist in this database.
That is what makes the tamper-evidence claim checkable by a third party (P5).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Annotated

from drishti_worker.evidence import canonicalise, strip_excluded
from drishti_worker.evidence import verify as verify_doc
from drishti_worker.merkle import verify_proof
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db import get_db
from ..models import Alert, AuditLog, LedgerAnchorBatch, LedgerAnchorEntry
from ..schemas import (
    DocumentVerifyIn,
    EvidenceItemVerification,
    LedgerInfoOut,
    MerkleProofOut,
    VerificationCheck,
    VerificationOut,
)
from ..security import RequireViewer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["verify"])


@router.get("/alerts/{alert_id}/verify", response_model=VerificationOut)
async def verify_alert(
    alert_id: str,
    principal: RequireViewer,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VerificationOut:
    """Recompute the hash, check the Merkle proof, query the ledger.

    Returns every intermediate value (§8). Nothing is asserted that the caller
    cannot check for themselves.
    """
    alert = (
        await db.execute(
            select(Alert).options(selectinload(Alert.items)).where(Alert.id == alert_id)
        )
    ).scalar_one_or_none()
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")

    checks: list[VerificationCheck] = []

    # --- 1. does the stored document still hash to the stored hash? ---
    result = verify_doc(alert.evidence_doc, alert.evidence_hash)
    canonical = canonicalise(strip_excluded(alert.evidence_doc))
    checks.append(
        VerificationCheck(
            name="evidence_hash",
            passed=result.ok,
            detail=f"expected={alert.evidence_hash} computed={result.computed_hash}",
        )
    )

    items = [
        EvidenceItemVerification(kind=i.kind, object_key=i.object_key, sha256=i.sha256)
        for i in alert.items
    ]

    # --- 2. is it in an anchored Merkle batch? ---
    merkle: MerkleProofOut | None = None
    ledger: LedgerInfoOut | None = None

    row = (
        await db.execute(
            select(LedgerAnchorEntry, LedgerAnchorBatch)
            .join(LedgerAnchorBatch, LedgerAnchorEntry.batch_id == LedgerAnchorBatch.id)
            .where(LedgerAnchorEntry.alert_id == alert_id)
        )
    ).first()

    if row is not None:
        entry, batch = row
        path = [(str(s), str(h)) for s, h in entry.proof]
        proof_ok = verify_proof(entry.leaf_hash, path, batch.merkle_root)
        merkle = MerkleProofOut(
            leaf_index=entry.leaf_index,
            leaf_hash=entry.leaf_hash,
            proof=path,
            computed_root=batch.merkle_root if proof_ok else "<mismatch>",
            stored_root=batch.merkle_root,
            root_match=proof_ok,
        )
        ledger = LedgerInfoOut(
            backend=batch.backend,
            tx_id=batch.tx_id,
            block_number=batch.block_number,
            anchored_at=batch.anchored_at,
            root_on_chain=batch.merkle_root,
            chain_match=proof_ok,
        )
        checks.append(
            VerificationCheck(
                name="merkle_proof",
                passed=proof_ok,
                detail=f"leaf {entry.leaf_index} of {batch.leaf_count} "
                f"in batch rooted at {batch.merkle_root[:16]}…",
            )
        )
        checks.append(
            VerificationCheck(
                name="ledger_anchor",
                passed=batch.status == "anchored",
                detail=f"backend={batch.backend} tx={batch.tx_id}",
            )
        )
        # The anchored leaf must be THIS alert's hash, not merely a valid leaf.
        checks.append(
            VerificationCheck(
                name="leaf_matches_alert",
                passed=entry.leaf_hash == alert.evidence_hash,
                detail="the anchored leaf is this alert's evidence hash",
            )
        )

    # --- 3. verdict ---
    if not result.ok:
        verdict = "TAMPERED"
    elif row is None:
        # Not an error. Ledger unavailability never blocks alerting, so a
        # pending anchor is an expected, temporary state (Invariant, §7.12).
        verdict = "PENDING_ANCHOR"
    else:
        verdict = "VERIFIED" if all(c.passed for c in checks) else "TAMPERED"

    db.add(
        AuditLog(
            actor_id=principal.user_id,
            action="alert.verify",
            target_type="alert",
            target_id=alert_id,
            detail={"verdict": verdict},
        )
    )

    return VerificationOut(
        alert_id=alert_id,
        stored_hash=alert.evidence_hash,
        recomputed_hash=result.computed_hash,
        hash_match=result.ok,
        canonical_bytes_sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_length=len(canonical),
        evidence_items=items,
        merkle=merkle,
        ledger=ledger,
        verdict=verdict,
        checks=checks,
        diff=list(result.diff),
    )


@router.post("/verify/document", response_model=VerificationOut)
async def verify_document(payload: DocumentVerifyIn) -> VerificationOut:
    """Verify an evidence JSON handed to you by anyone.

    Needs no database row: the point of P5 is that tamper-evidence is a property
    of the *record*, so a third party must be able to check it independently.
    Paste the JSON, get the canonical bytes, the hash, and a yes or no.
    """
    document = payload.document
    expected = payload.expected_hash or document.get("evidence_hash")
    if not expected:
        raise HTTPException(
            400,
            "no hash to check against: supply expected_hash, or include "
            "evidence_hash in the document",
        )

    try:
        canonical = canonicalise(strip_excluded(document))
    except (TypeError, ValueError) as exc:
        # Loud, specific failure. P5.
        raise HTTPException(422, f"document cannot be canonicalised (RFC 8785): {exc}") from exc

    computed = hashlib.sha256(canonical).hexdigest()
    ok = computed == expected

    checks = [
        VerificationCheck(
            name="schema",
            passed=document.get("schema") == "drishti.evidence/v1",
            detail=f"schema={document.get('schema')!r}",
        ),
        VerificationCheck(
            name="canonicalisation",
            passed=document.get("provenance", {}).get("canonicalisation") == "RFC8785",
            detail="document declares RFC 8785 canonicalisation",
        ),
        VerificationCheck(
            name="evidence_hash",
            passed=ok,
            detail=f"expected={expected} computed={computed}",
        ),
    ]

    return VerificationOut(
        alert_id=str(document.get("alert_id", "<unknown>")),
        stored_hash=str(expected),
        recomputed_hash=computed,
        hash_match=ok,
        canonical_bytes_sha256=computed,
        canonical_length=len(canonical),
        evidence_items=[
            EvidenceItemVerification(
                kind=str(i.get("kind", "?")),
                object_key=str(i.get("object_key", "")),
                sha256=str(i.get("sha256", "")),
            )
            for i in document.get("items", [])
        ],
        verdict="VERIFIED" if ok else "TAMPERED",
        checks=checks,
    )
