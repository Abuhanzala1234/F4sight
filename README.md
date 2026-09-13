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
| **Face matching** | Interface, config, DB schema and privacy invariants exist; `faces.py` itself (§7.10) is **not yet implemented** — see Gate 9 |
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

### Gate 2 — Model weights · **CLOSED**

`make models` has been run. YOLO11n (11 MB) and YOLO11s (39 MB) are downloaded and exported to ONNX (opset 12, dynamic batch), and the two InsightFace models are extracted from the InsightFace bundle. `models/MANIFEST.json` verifies.

Two things were fixed in the process:

- The fetcher saved the 127 MB InsightFace **zip** under the name `scrfd_500m.onnx` and recorded its hash in the manifest. The integrity check passed happily, because a SHA-256 tells you a file has not changed, not that it is the file it claims to be. It now extracts `det_500m.onnx` and `w600k_mbf.onnx` properly and leaves the gender/age and landmark nets in the archive, unshipped.
- Every `read_text`/`write_text` in `scripts/` now passes `encoding="utf-8"`. Python defaults to the locale encoding, which is cp1252 on Windows, so `make bench` crashed writing its own report the moment the table contained a `→`.

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

### Gate 5 — A GPU · **CLOSED (CUDA), one sliver open (TensorRT)**

The `bop` profile now runs on a real NVIDIA GPU. Measured on an **RTX 3050 Laptop (4 GB, compute 8.6, driver 616.92)** with YOLO11s at 640×640 — every number below is from `make bench`, in [`docs/BENCH.md`](docs/BENCH.md):

| | batch=1 | batch=8, per frame |
| --- | ---: | ---: |
| CPU (Ryzen, ORT CPU EP) | 40.1 ms | 46.7 ms |
| **CUDA EP** | **15.2 ms** | **12.9 ms → 76 fps aggregate** |

**6.4 cameras at the profile's 12 fps**, on a laptop GPU a tier *below* the RTX 3060 the profile targets. The §5 claim of 8–12 cameras is no longer theoretical — it is bracketed from below by measured hardware.

Three things had to be fixed to get here, and all three would have bitten a real deployment:

1. **`fp16: true` was decorative.** It was parsed from config and never passed to a provider, so the profile that advertised FP16 ran FP32. It is now `trt_fp16_enable`.
2. **The `bop` profile pointed at `yolo11s.trt`,** a serialised TensorRT engine. `ort.InferenceSession` cannot load one — it takes ONNX and builds the engine itself. Any attempt to run this profile would have failed at startup.
3. **CUDA bound to nothing, silently.** The CUDA libraries ship as `nvidia-*` wheels that unpack somewhere Windows does not search for DLLs, so ORT reported a missing `cublasLt64_13.dll`, fell back to CPU, and ran perfectly — ten times slower than the hardware allows, with no obvious symptom. `onnxruntime.preload_dlls()` now runs before session creation.

A fourth fix was found by measuring rather than by reading: with the GPU doing inference in 5.4 ms, **frame preprocessing became the bottleneck at 5.8 ms/frame** — three full-array temporaries and a fresh 39 MB allocation per batch. Rewritten to fill a reused NCHW buffer, verified bit-identical to the implementation it replaced. Batched throughput went from 53 to 90 fps on YOLO11n; that is where most of the table above comes from.

**Still open — TensorRT EP.** Not a hardware problem, a packaging one: ONNX Runtime 1.30 links `nvinfer_10.dll` (TensorRT 10), and TensorRT 10 publishes no wheels for Python 3.14 — only TensorRT 11, which ORT does not yet link. So the ceiling above is CUDA's, not TensorRT's, and TensorRT FP16 would typically add another 1.3–2×.

- **Give me:** Python 3.12 or 3.13 (where `tensorrt-cu13==10.x` installs), and `make bench PROFILE=bop` produces the TensorRT row with no code change — the provider list already prefers it.
- **Fallback in place, and exercised:** the provider chain falls back TensorRT → CUDA → CPU on its own. Both fallbacks were hit for real on this machine and are visible in the run log, not just in a unit test.

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

### Gate 9 — Faces (§7.10) is not implemented · *blocks Phase 10*

Everything **around** face matching exists and is correct: `config/faces.yaml`
ships `enabled: false` and the module contract refuses to load weights while
it is; the `watchlist_person`/`watchlist_face_vector` tables and their pgvector
HNSW index are migrated; `WatchlistFaceRule` in `rules.py` is written, wired to
`ctx.face_hit`, and gated on `faces.enabled`; the API's `/watchlist/persons`
routes 403 while disabled. The two ONNX models (`scrfd_500m.onnx`,
`w600k_mbf.onnx`) are downloaded and verified in the manifest. What is missing
is `faces.py` itself — `detect_faces`, `embed`, `match`, `WatchlistIndex` — and
the pipeline wiring that would call it (the ANPR equivalent of this, plate
reading → voting → watchlist match → `plate_hit`, is now built; faces never
got its counterpart).

This is a real, scoped gap, not an oversight papered over: it is Phase 10 in
§16, behind Phases 0–9 which are complete, and P6 makes it the one module
where getting the invariant wrong (a non-matching embedding must never persist)
is worse than not shipping it under time pressure.

- **Give me:** time for one more pass, or say which is more valuable — this,
  or spending that time elsewhere. It is disabled by default either way, so
  its absence changes nothing about what `make demo` shows today.
- **Risk if skipped:** the "face matching" row in §4's free-stack table and
  the demo script cannot claim a working face-watchlist hit. Nothing else
  depends on it — `faces.enabled: false` means the rest of the pipeline never
  calls into it regardless.

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
worker/src/drishti_worker/  the analytics pipeline (23 modules)
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

**384 Python tests + 6 TypeScript tests**, all passing with no infrastructure required —
checked on every push by [GitHub Actions](.github/workflows/ci.yml), which runs the
identical `make test`/`make lint` commands rather than a separate CI-only path.

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
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Incident playbooks — what to actually type when something's wrong |

---

## Team

**SW-73 — ByteForge.** *(Add member names and roles before submission — see Gate 7.)*
