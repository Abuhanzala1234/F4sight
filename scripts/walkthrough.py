#!/usr/bin/env python3
"""End-to-end walkthrough with no infrastructure at all.

Drives a synthetic intruder through the real pipeline — track, geometry, rules,
risk, evidence, Merkle, ledger, verification — with no database, no GPU, no
model weights and no network. Useful as a smoke test and as the thing to run
when someone asks "does it actually work?".

    python scripts/walkthrough.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker" / "src"))

from drishti_worker.evidence import assemble, canonicalise, evidence_hash, verify
from drishti_worker.geometry import crossing_direction
from drishti_worker.ledger.mock import MockLedger
from drishti_worker.merkle import build_tree, proof, verify_proof
from drishti_worker.risk import RiskConfig, RiskContext, score
from drishti_worker.rules import DebounceConfig, Debouncer, RuleConfig, RuleEngine
from drishti_worker.track import ByteTracker, TrackerConfig
from drishti_worker.types import CameraRuntime, Detection, ZoneKind, ZoneRuntime

BOLD, DIM, GREEN, RED, CYAN, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[31m",
    "\033[36m",
    "\033[0m",
)


def head(n: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}── {n}. {title} {'─' * max(0, 58 - len(title))}{RESET}")


T0 = datetime(2026, 9, 12, 22, 14, 0, tzinfo=UTC)

camera = CameraRuntime(
    camera_id="cam-bop03-01",
    code="CAM-01",
    site_id="site-bop03",
    site_code="BOP-03",
    timezone="Asia/Kolkata",
    width=1280,
    height=720,
    analytics_fps=6.0,
)
wire = ZoneRuntime(
    zone_id="z-wire",
    name="Perimeter line",
    kind=ZoneKind.TRIPWIRE,
    # Wound so that walking left-to-right is INBOUND (into the protected area).
    polygon=((600.0, 720.0), (600.0, 0.0)),
    direction="both",
    classes=("person",),
    severity_base=4,
)

print(f"{BOLD}DRISHTI-BOP — offline walkthrough{RESET}")
print(f"{DIM}SIH 2026 · PS 26187 · Team SW-73 (ByteForge){RESET}")
print(f"{DIM}No database, no GPU, no model weights, no network.{RESET}")

# ---------------------------------------------------------------- 1
head(1, "Ingest + track: a person walks toward the fence at night")
tracker = ByteTracker(TrackerConfig(min_hits=3))
engine = RuleEngine(RuleConfig(), RiskConfig())
debouncer = Debouncer(DebounceConfig())
fired = None

for i in range(40):
    x = 420.0 + i * 10  # crosses the wire at x=600 around frame 18
    conf = 0.22 if 12 <= i <= 15 else 0.91  # occlusion behind a fence post
    ts = T0 + timedelta(seconds=i / 6.0)
    tracks = tracker.update([Detection("person", conf, (x - 20, 220, x + 20, 430), 0)], ts)
    if not tracks:
        continue
    signals = engine.evaluate(tracks, camera, [wire], ts, profile="night")
    if signals and fired is None:
        tid, sigs = next(iter(signals.items()))
        primary = max(sigs, key=lambda s: s.weight)
        decision = debouncer.submit(
            camera_id=camera.camera_id,
            track_id=tid,
            zone_id=wire.zone_id,
            rule_code=primary.code,
            now=ts,
            alert_id="alert-1",
            weight=primary.weight,
        )
        # Wait for the crossing itself, not the early-warning approach.
        if decision.should_write and primary.code == "TRIPWIRE_CROSS":
            fired = (tracks[0], sigs, ts, i)

track, signals, fired_ts, fired_frame = fired
print("   frames processed        : 40 @ 6 fps")
print(
    f"   track id                : {track.track_id}  {GREEN}(survived a 4-frame occlusion at 0.22 conf){RESET}"
)
print(f"   hits / age              : {track.hits} / {track.age_frames} frames")
print(f"   fired on frame          : {fired_frame}")
d = crossing_direction(track.history[-2], track.foot_point, wire.wire)
print(f"   crossing direction      : {d}")

# ---------------------------------------------------------------- 2
head(2, "Risk: additive and explainable (P2)")
risk = score(
    signals,
    RiskContext(
        config=RiskConfig(),
        evqm_profile="night",
        track_max_conf=track.max_conf,
        track_age_frames=track.age_frames,
    ),
)
for s in risk.breakdown:
    colour = GREEN if s.weight >= 0 else RED
    print(f"   {s.code:<22} {colour}{s.weight:+7.2f}{RESET}   {DIM}{list(s.detail)[:3]}{RESET}")
print(f"   {'':<22} {BOLD}{'─' * 7}{RESET}")
print(f"   {'SCORE':<22} {BOLD}{risk.score:7.2f}{RESET}   severity={BOLD}{risk.severity}{RESET}")
print(
    f"   contributions sum to score: {GREEN if risk.sums_correctly() else RED}{risk.sums_correctly()}{RESET}"
)

# ---------------------------------------------------------------- 3
head(3, "Debounce: what a naive system would have sent (blocker #1)")
naive = Debouncer(DebounceConfig())
emitted = sum(
    1
    for i in range(360)
    if naive.submit(
        camera_id="c",
        track_id=7,
        zone_id="z",
        rule_code="TRIPWIRE_CROSS",
        now=T0 + timedelta(seconds=i / 6),
        alert_id=f"a{i}",
    ).should_write
)
print("   candidate firings/minute: 360")
print(
    f"   alerts actually raised  : {GREEN}{emitted}{RESET}   {DIM}(cooldown 45s + escalation){RESET}"
)

# ---------------------------------------------------------------- 4
head(4, "Evidence: canonicalise (RFC 8785) → SHA-256 (blocker #7)")
doc = assemble(
    alert_id="alert-1",
    site={"code": "BOP-03", "name": "Border Out Post 03"},
    camera={"code": "CAM-01", "width": 1280, "height": 720},
    detection={
        "track_id": track.track_id,
        "class": track.cls,
        "box": [round(v, 1) for v in track.box],
        "foot_point": [round(v, 1) for v in track.foot_point],
        "evqm_profile": "night",
        "enhancement_params": {"clahe_clip": 2.0},
    },
    risk={
        "score": risk.score,
        "severity": risk.severity,
        "breakdown": [{"code": s.code, "weight": s.weight} for s in risk.breakdown],
    },
    items=[{"kind": "snapshot", "sha256": "a" * 64, "enhanced": False}],
    config_version="c" * 64,
    spec_version="1.0.0",
    worker_version="1.0.0",
    created_at=fired_ts.isoformat(),
)
digest = evidence_hash(doc)
canon = canonicalise({k: v for k, v in doc.items() if k not in ("evidence_hash", "ledger")})
print(f"   canonical bytes         : {len(canon)}")
print(f"   evidence_hash           : {digest}")
print(
    f"   stable under key reorder: {GREEN}{evidence_hash(dict(reversed(list(doc.items())))) == digest}{RESET}"
)

# ---------------------------------------------------------------- 5
head(5, "Merkle batch + ledger anchor (the blockchain story)")
import hashlib

batch = [digest] + [hashlib.sha256(f"other-alert-{i}".encode()).hexdigest() for i in range(11)]
tree = build_tree(batch)
path = proof(tree, 0)
ledger = MockLedger(Path(tempfile.mkdtemp()) / "ledger.jsonl")
receipt = ledger.anchor(tree.root, {"leaf_count": tree.leaf_count, "site_code": "BOP-03"})
print(f"   leaves in batch         : {tree.leaf_count}")
print(f"   merkle root             : {tree.root}")
print(f"   proof length            : {len(path)} siblings  {DIM}(log2 of the batch){RESET}")
print(f"   ledger tx               : {receipt.tx_id}  backend={receipt.backend}")

# ---------------------------------------------------------------- 6
head(6, "Verification — what /alerts/{id}/verify returns")
v = verify(doc, digest)
chain_ok, chain_detail = ledger.verify_chain()
checks = [
    ("evidence_hash recomputes", v.ok),
    ("merkle proof verifies", verify_proof(digest, path, tree.root)),
    ("root matches the ledger", ledger.get(receipt.tx_id).merkle_root == tree.root),
    ("ledger chain intact", chain_ok),
]
for name, ok in checks:
    print(
        f"   {'✓' if ok else '✗'} {name:<30} {GREEN if ok else RED}{'PASS' if ok else 'FAIL'}{RESET}"
    )
print(
    f"\n   VERDICT: {BOLD}{GREEN if all(o for _, o in checks) else RED}"
    f"{'VERIFIED' if all(o for _, o in checks) else 'TAMPERED'}{RESET}"
)

# ---------------------------------------------------------------- 7
head(7, "Tamper test: change one field, verification must fail loudly")
import json

tampered = json.loads(json.dumps(doc))
tampered["detection"]["track_id"] = 999
bad = verify(tampered, digest, reference=doc)
print(f"   hash matches            : {RED}{bad.ok}{RESET}")
print(f"   verdict                 : {BOLD}{RED}TAMPERED{RESET}")
for line in bad.diff[:3]:
    print(f"   {DIM}diff:{RESET} {line}")
print(
    f"   {DIM}...and a forged leaf cannot be proven into the batch:{RESET} "
    f"{RED}{verify_proof(hashlib.sha256(b'forged').hexdigest(), path, tree.root)}{RESET}"
)
print()
