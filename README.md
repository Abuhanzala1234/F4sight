# DRISHTI-BOP

**AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure**

Smart India Hackathon 2026 · Problem Statement **26187** · Theme: Blockchain & Cybersecurity · Team **SW-73 (ByteForge)**

---

Border Out Posts already have CCTV. Those cameras do two things: show live video, and record it. Everything else is a human in a room, and the failure mode of a tired human is silent — nobody knows which frames were missed.

DRISHTI-BOP is software that bolts onto **the RTSP streams that already exist**. It detects, tracks, and reasons about what it sees; it scores every candidate event with an additive model whose terms are all visible; it raises debounced, evidence-backed alerts; and it writes a tamper-evident audit trail anchored to a permissioned ledger.

It runs offline, on one machine, entirely on free and open-source parts. **No API key, no cloud account, no metered inference, ₹0 marginal cost.**

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Hard gates — what this repo cannot do for itself](#hard-gates)
- [Architecture](#architecture)
- [The free stack](#the-free-stack)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Documentation](#documentation)

---

## What it does

| Capability | How |
| --- | --- |
| **Intrusion detection** | Polygon zones and directional tripwires on the image plane, evaluated against each track's foot-point |
| **Object detection & tracking** | YOLO11 (ONNX) + ByteTrack reimplemented in-repo — the low-confidence second pass keeps an ID alive through the occlusion behind a fence post |
| **ANPR** | Two-stage plate detect + OCR with **multi-frame voting**; the database stores an HMAC, never a plate |
| **Face matching** | InsightFace, opt-in, **disabled by default**, non-matching embeddings destroyed at track close |
| **Explainable risk** | Every alert carries an additive breakdown whose terms sum exactly to the score |
| **Tamper-evident evidence** | RFC 8785 canonicalisation → SHA-256 → Merkle batch → Hyperledger Fabric |
| **Environmental adaptation** | EVQM measures brightness, contrast, blur, fog, noise and picks a processing profile, with hysteresis so a passing headlight cannot flip it |
| **Operator console** | Live HLS wall, keyboard-first triage, risk waterfall, verification panel |

### Three things it deliberately does *not* do

1. **No automated response.** No barrier, siren, or dispatch. The system recommends; a human adjudicates.
2. **No biometric enrolment.** Faces are matched against an explicitly curated watchlist or not at all.
3. **No plaintext plate storage.** A database leak must not be a list of who drove where.

---

## Quick start

**Prerequisites:** Docker + Docker Compose, Python 3.11+, Node 20+.

```bash
git clone <this repo> && cd f4sight

make install     # python venv + npm install
make up          # postgres + minio + redis + mediamtx
make migrate     # create the schema
make seed        # demo site, 3 cameras, zones, users, watchlist
make models      # download the free model weights (~166 MB, once)
make fixtures    # synthesise sample footage
make demo        # worker + api + anchor, all together
```

Then open **http://localhost:8000/**. `make seed` prints the demo credentials.

### See it work with no infrastructure at all

If you have neither Docker nor the model weights, this still runs and proves the whole evidence chain:

```bash
python scripts/walkthrough.py
```

It drives a synthetic intruder through the real pipeline — track, geometry, rules, risk, evidence hashing, Merkle proof, ledger anchor, verification, and a tamper test — in about a second.

```
── 2. Risk: additive and explainable (P2) ───────────────────
   TRIPWIRE_CROSS          +45.00
   NIGHT_MOVEMENT          +20.00
                           ───────
   SCORE                     65.00   severity=high
   contributions sum to score: True
```

---

## Hard gates

**Everything below is something this repository cannot produce for itself.** Each one is a real external dependency — hardware, footage, credentials, or a decision. The system is built so that *none of them block the demo*: there is a working fallback for every single item. But the fallback is not the real thing, and here is exactly what the real thing needs.

### Gate 1 — Docker · *blocks `make up`, `make demo`*

Not installed on the machine this was built on, so Postgres, MinIO, Redis and MediaMTX have never actually been started.

- **Give me:** Docker Desktop installed, or run `make up` yourself and paste the output.
- **Fallback in place:** every DB-free path is tested (341 tests pass with no infrastructure), and the worker falls back to fixture cameras defined in config when the database is unreachable.
- **Risk if skipped:** migrations, seed, the live wall and the anchor service have never run end to end. This is the **single highest-value gate** — everything else is downstream of it.

### Gate 2 — Model weights · *blocks real detection*

`make models` needs network access to download YOLO11n (~10 MB) from GitHub, and `pip install ultralytics` (build-time only) to export it to ONNX.

- **Give me:** run `make models` on a connected machine, or confirm it is fine to `pip install ultralytics` here.
- **Fallback in place:** `detector.backend: mock` produces scripted detections; every pipeline stage downstream is exercised and tested against it.
- **Risk if skipped:** no real accuracy numbers, and `make bench` reports inference at 0 ms because it is benchmarking the mock.

### Gate 3 — Real labelled footage · *blocks `make eval`*

The evaluation set (§15) needs ≥ 40 clips with ground truth: day, dusk, night, fog, rain-on-lens, crowd, vehicle, **empty scenes**, and two deliberate tamper clips.

- **Give me:** clips in `eval/clips/` plus `eval/labels.jsonl`. Even 10 clips would let me report honest numbers. Border-adjacent CCTV, campus perimeter footage, or any fixed-camera outdoor video works.
- **Fallback in place:** `make fixtures` synthesises three clips with OpenCV — enough to prove the pipeline, not enough to measure it.
- **Risk if skipped:** **no defensible precision/recall figures.** The targets in §14 stay aspirations. A judge asking "how accurate is it?" gets an honest "we have not measured it on real footage", which is a weak answer.

### Gate 4 — A real RTSP camera · *blocks the P1 claim*

The entire premise is "works with existing CCTV". That has only been tested against MediaMTX replaying MP4s.

- **Give me:** one RTSP URL (any IP camera, even a phone running an RTSP server app), or confirmation you have tested against one.
- **Fallback in place:** the watchdog and reconnect logic is unit-tested, and fixture streams exercise the same code path.
- **Risk if skipped:** real cameras have quirks — H.264 baseline, B-frames, credentials in the URL, ONVIF discovery, 4CIF resolutions — that fixtures never reproduce.

### Gate 5 — A GPU · *blocks the `bop` profile*

The `bop` profile targets an RTX 3060+ with TensorRT FP16. This machine is CPU-only (Apple silicon, CoreML EP available).

- **Give me:** access to a GPU box, or accept that the `bop` numbers stay theoretical.
- **Fallback in place:** the `laptop` profile is the default and works on CPU; ONNX Runtime picks the best available provider automatically.
- **Risk if skipped:** the 8–12 camera claim in §5 is unverified. The 1–2 camera laptop claim is real.

### Gate 6 — Hyperledger Fabric · *blocks the real ledger backend*

`make fabric-up` needs Docker plus `fabric-samples` (~2 GB) and Go.

- **Give me:** Docker + `./install-fabric.sh docker samples binary`, then `make fabric-up`.
- **Fallback in place:** the mock ledger is a **real append-only hash-chained log** that detects tampering and names the broken entry. It exercises the identical code path; switching is one config line.
- **Risk if skipped:** the word "blockchain" in the problem statement is answered by a chained JSONL file rather than Fabric. Defensible, but weaker in the room.

### Gate 7 — Decisions only you can make

| Decision | Default I chose | Why you might change it |
| --- | --- | --- |
| **AGPL vs Apache** | YOLO11 (AGPL-3.0) | If the deploying org rejects AGPL, set `detector.backend: rtdetr` — Apache-2.0, ~2 points mAP lower, no other change |
| **Face analytics** | Disabled | It is the answer to the privacy question. Turning it on is a policy decision, not an engineering one |
| **Team members** | Not listed | Add real names/roles to this README before submission |
| **Site geometry** | `BOP-03`, Delhi coordinates, invented zones | Replace with your actual demo site in `config/sites/` |

### Gate 8 — Things I could not visually verify

The dashboard **compiles, typechecks in strict mode, and builds** (763 KB bundle, fonts inlined for offline use), and its logic is unit-tested. But I have no browser here, so **nobody has looked at it.** Spacing, contrast at projector gamma, and whether the risk waterfall actually reads at a glance are unverified.

- **Give me:** `make dash`, then a screenshot or a list of what looks wrong.

---

## Architecture

```
 Existing IP CCTV ──RTSP──▶ MediaMTX ──┬── recording (survives an analytics crash, P8)
                                       ├── HLS ──▶ browser (RTSP is unplayable in one)
                                       └── RTSP ──▶ drishti-worker
                                                      │
        ingest → EVQM → letterbox → enhance → detect → track
              → geometry → rules → risk → debounce → evidence → sinks
                                                      │
              ┌───────────────────────────────────────┼──────────────┐
              ▼                    ▼                  ▼              ▼
          Postgres              MinIO              Redis      drishti-anchor
          +pgvector          snapshots/clips      streams      Merkle → Fabric
              │                    │                  │
              └────────────▶ drishti-api ◀────────────┘
                          REST · WebSocket · verify
                                   │
                          React command center
```

Four processes, three stateful services, one machine. See [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md) §3.

### Design decisions worth knowing

- **The worker is threaded, not async.** The hot path is CPU-bound and every heavy call releases the GIL. One process, not one per camera, because the accelerator is a shared resource.
- **The frame queue drops the oldest.** On a slow host we analyse *recent* frames. An alert about where someone was thirty seconds ago is not an alert.
- **Evidence is the original frame.** Enhancement is for the model; evidence is for the court. Enhancement parameters are stored as metadata instead.
- **Anchoring never blocks alerting.** `ledger_status='pending'` is a perfectly good alert.

---

## The free stack

| Layer | Choice | Licence |
| --- | --- | --- |
| Detection | YOLO11n/s ONNX · RT-DETR-R18 fallback | AGPL-3.0 · Apache-2.0 |
| Tracking | ByteTrack, reimplemented in-repo | MIT (algorithm) |
| ANPR | YOLO11n plate + PaddleOCR v4 | AGPL-3.0 · Apache-2.0 |
| Faces (opt-in) | InsightFace SCRFD + ArcFace | MIT |
| Inference | ONNX Runtime (+TensorRT / CoreML) | MIT |
| Database | PostgreSQL 16 + pgvector | PostgreSQL Licence |
| Object store | MinIO | AGPL-3.0 |
| Media | MediaMTX | MIT |
| Ledger | Hyperledger Fabric 2.5 | Apache-2.0 |
| API | FastAPI | MIT |
| Dashboard | React + Vite + Tailwind | MIT |

Full provenance, licence position and upgrade paths: [`docs/MODELS.md`](docs/MODELS.md).

---

## Repository layout

```
CLAUDE.md                   working rules for this repository
docs/BUILD_SPEC.md          authoritative spec — §7 contracts are frozen
docs/MODELS.md              every model: source, licence, size, cost
docs/BENCH.md               generated by `make bench`
config/                     every tunable number — none in source
worker/src/drishti_worker/  the analytics pipeline (22 modules)
api/src/drishti_api/        FastAPI: REST, WebSocket, verification
dashboard/src/              React command center
fabric/chaincode/           evidencecc — stores Merkle roots, nothing else
infra/                      compose, MediaMTX, Postgres bootstrap
scripts/                    models, fixtures, bench, eval, demo, walkthrough
```

---

## Testing

```bash
make test        # pytest + vitest
make lint        # ruff + black + mypy + tsc
make bench       # → docs/BENCH.md
```

**341 Python tests + 6 TypeScript tests**, all passing with no infrastructure required.

Geometry, risk scoring, Merkle proofs, JCS canonicalisation and plate validation get **property-based tests** (`hypothesis`) — they are pure functions with sharp edge cases:

- Merkle proofs verify for **every tree size from 1 to 257**, and forged leaves never verify
- Risk contributions **always** sum to the score, over 400 generated signal combinations
- The JCS canonicaliser passes the RFC 8785 number vectors (`1e21`, `1e-7`, `-0`, `5e-324`)
- Point-in-polygon is invariant under translation and rotation, and deterministic on edges

We mock the GPU, Fabric and the network. Never the logic.

---

## Documentation

| Document | What it is for |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | Working rules, invariants, blockers |
| [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md) | The authoritative spec — architecture, data model, frozen module contracts, build phases |
| [`docs/MODELS.md`](docs/MODELS.md) | Every weight file and its licence |
| [`docs/BENCH.md`](docs/BENCH.md) | Measured performance, regenerated by `make bench` |

---

## Team

**SW-73 — ByteForge.** *(Add member names and roles before submission — see Gate 7.)*
