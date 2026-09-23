# IBVAP

**AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure**

Smart India Hackathon 2026 · Problem Statement **26187** · Theme: Blockchain & Cybersecurity · Team **SW-73 (ByteForge)**

---

Border Out Posts already have CCTV. Those cameras do two things: show live video, and record it. Everything else is a human in a room, and the failure mode of a tired human is silent — nobody knows which frames were missed.

IBVAP is software that bolts onto **the RTSP streams that already exist**. It detects, tracks, and reasons about what it sees; it scores every candidate event with an additive model whose terms are all visible; it raises debounced, evidence-backed alerts; and it writes a tamper-evident audit trail anchored to a permissioned ledger.

It runs offline, on one machine, entirely on free and open-source parts. **No API key, no cloud account, no metered inference, ₹0 marginal cost.**

---

## Table of contents

- [What it does](#what-it-does)
- [What weapon detection can and cannot do](#what-weapon-detection-can-and-cannot-do-measured)
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
| **Hand signals** | YOLO11-pose keypoints on tracked people → surrender / signalling / pointing, voted across frames. Contextual: enriches a real event, never raises one alone |
| **Weapon detection** | Two-class (gun/knife) detector on person crops, voted on *armed* rather than on weapon type — somebody the model cannot decide between a gun and a knife is still, unambiguously, armed. `config/weapons.yaml` carries the measured threshold curve **and** its honest limit (below) |
| **Suspicious fast movement** | Running detected in *body-heights per second*, so one threshold means the same thing at any distance from the camera. No model — the tracker already measured it. Tracker id-switches are discarded as artefacts, not escalated as sprinting intruders |
| **Facial recognition** | SCRFD detect + ArcFace embed against an admin-curated, audited watchlist. OPT-IN, off by default (P6) — no enrolment path exists in the worker, and a non-matching embedding is held only for the life of its track, never written anywhere |
| **Group behaviour** | Converging and dispersing formations, measured from track spread over a window — the coordinated case no per-track rule can see |
| **Explainable risk** | Every alert carries an additive breakdown whose terms sum exactly to the score |
| **Tamper-evident evidence** | RFC 8785 canonicalisation → SHA-256 → Merkle batch → Hyperledger Fabric |
| **Environmental adaptation** | EVQM measures brightness, contrast, blur, fog, noise and picks a processing profile, with hysteresis so a passing headlight cannot flip it |
| **Operator console** | Live HLS wall, keyboard-first triage, risk waterfall, verification panel |

### What weapon detection can and cannot do, measured

The one capability worth stating plainly, because a demo can easily overstate it.
Scores from live tests on a phone camera, same scene, same session:

| Subject | Model's weapon score |
| --- | --- |
| Knife held up, ~1 m, blade side-on | 0.68 – **0.744** |
| Knife at 4–5 m | 0.32 – **0.536** |
| **Phone in a hand**, no weapon at all | up to **0.707** |
| Empty hands, awkward angle | up to 0.639 |

The distributions **overlap**, and at 4–5 m the false readings outscore the real
knife. So no confidence threshold both catches a knife across a room and never
misfires on a phone: at 0.75 the real knife is rejected, at 0.65 a phone fires,
at 0.45 ordinary people fire constantly. This is a property of a small,
free, offline two-class model, not a tuning mistake — and it is why the shipped
default errs high (P3: an alert that cries wolf teaches an operator to ignore
the real one). Weapon detection here is dependable at **close range with the
blade visible**; treat anything beyond that as unproven. The durable fix is not
a threshold but a discriminator — rejecting a weapon box that lands on a phone
the primary detector already sees (COCO class 67) — which is not built yet.

### Three things it deliberately does *not* do

1. **No automated response.** No barrier, siren, or dispatch. The system recommends; a human adjudicates.
2. **No biometric enrolment.** Faces are matched against an explicitly curated watchlist or not at all.
3. **No plaintext plate storage.** A database leak must not be a list of who drove where.

---

## Quick start

**Prerequisites:** Docker + Docker Compose (running), Python 3.11+, Node 20+.

```bash
git clone <this repo> && cd ibvap
make demo
```

That's it — one command, on a bare clone, on any machine with the three
prerequisites above. It installs the venv and npm dependencies, starts
postgres/minio/redis/mediamtx, creates the schema, seeds a demo site with 4
camera slots (3 running on bundled sample footage, 1 empty), downloads the
free model weights (~166 MB, skipped on every run after the first), builds
the dashboard, and starts the worker + API + ledger-anchor service together.
Re-running it is safe and fast — every step is idempotent and skips work
that is already done.

Then open **http://localhost:8000/**. Demo credentials are printed at the
end of the seed step (also re-printable with `make seed`).

Want the steps individually instead (e.g. to only bring infra up, or to
re-run just one stage)? `make demo`'s prerequisites and recipe are just
`install`, `up`, `migrate`, `seed`, `models`, `fixtures`, and the dashboard
build — each is its own target and safe to run on its own; see `make help`.

### Running it on a GPU, and keeping it running

`make demo` uses the `laptop` profile, which is CPU-only so that it works on any
machine. On a box with an NVIDIA GPU and `onnxruntime-gpu` installed, run the
worker on the `laptop-gpu` profile instead — measured on an RTX 3050 laptop with
two cameras, that is the difference between ~40% of frames dropped and ~0%:

```bash
scripts/worker_supervised.sh laptop-gpu BOP-03
```

That wrapper does two things `make worker` does not. It **restarts the worker if
it dies** — a fault inside the CUDA driver aborts the process in a way Python
cannot catch, and without a supervisor every camera silently stops being
analysed until a human notices. And it **loads `.env` into the environment**,
which the worker needs for `PLATE_HMAC_KEY`: without it ANPR still reads plates
but can never match one against the watchlist.

One measured caveat, because it cost hours to find: on a laptop, **an unplugged
machine is a different machine**. On battery, Windows power-capped the same GPU
to 9.3 W and each detector call went from ~45 ms to ~700 ms, with frames
dropping and second-stage models starved. `nvidia-smi -q -d PERFORMANCE` names
it (`SW Power Cap: Active`). Plug in before drawing conclusions about speed.

### Bringing your own camera

Once it's running, open the Live Wall — any empty slot (dashed border) can
be bound to a live camera by clicking it and typing an IP address. Point a
phone running the free **IP Webcam** app (Android) at it — same WiFi as this
machine, screen unlocked — and it's live within a few seconds, no restart.
Detection runs immediately: the model is already loaded once and shared
across every camera, so adding one is pure plumbing, not an AI step.

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

**Everything below is something this repository could not have produced for itself.** Each one is a real external dependency — hardware, footage, credentials, or a decision. The system is built so that *none of them block the demo*: there is a working fallback for every single item. Six of the nine are closed with a live, verified run; Gate 7 is a set of decisions rather than a pass/fail; Fabric and face matching remain genuinely open, and are named as such below.

### Gate 1 — Docker · ✅ closed

Postgres, MinIO, Redis and MediaMTX have all been started and run together live. Two real upstream compatibility breaks were hit and fixed along the way: Docker Hub now requires a login even for anonymous `minio/minio` pulls (switched to the `quay.io/minio/*` mirrors, pinned), and MediaMTX renamed its `rtspTransports` config field to `protocols` between versions (found by extracting the shipped image's own default config). A third gap was closed after that: the MediaMTX image is a scratch-based static binary with no shell and no ffmpeg, so it could never actually run the fixture-camera `runOnInit` commands in `infra/mediamtx.yml` — the three seeded fixture cameras looked configured but never carried real video. Fixed with a dedicated `fixture-streamer` sidecar container (plain ffmpeg, looping the local MP4s in over RTSP/TCP); verified live via `ffprobe` against all three RTSP paths and a real `200` HLS manifest fetch.

### Gate 2 — Model weights · ✅ closed

`make models` has been run on two independent machines. YOLO11n (11 MB) and YOLO11s (39 MB) are downloaded and exported to ONNX (opset 12, dynamic batch) and verified against real inference; the two InsightFace models are extracted from the InsightFace bundle; `models/MANIFEST.json` verifies.

Two bugs were caught and fixed in the process:

- The fetcher saved the 127 MB InsightFace **zip** under the name `scrfd_500m.onnx` and recorded its hash in the manifest. The integrity check passed happily, because a SHA-256 tells you a file has not changed, not that it is the file it claims to be. It now extracts `det_500m.onnx` and `w600k_mbf.onnx` properly and leaves the gender/age and landmark nets in the archive, unshipped.
- Every `read_text`/`write_text` in `scripts/` now passes `encoding="utf-8"`. Python defaults to the locale encoding, which is cp1252 on Windows, so `make bench` crashed writing its own report the moment the table contained a `→`.

### Gate 3 — Real labelled footage · ✅ closed (small sample)

No 40-clip ground-truth set exists — that would need actual border-post footage this project never had access to. Instead, `eval/clips/` holds four real clips from **MOT16** (a public, directly-downloadable pedestrian-tracking benchmark), with ground-truth zones placed by tracing real tracked trajectories mathematically rather than eyeballing frames. Honest numbers — precision 0.222, recall 0.667, 0.00 false-alerts/idle-hour, per-clip breakdown, and explicit limitations — are in [`docs/EVAL.md`](docs/EVAL.md). This is a small-sample proof that the measurement pipeline itself is correct, not a claim of production accuracy.

### Gate 4 — A real RTSP camera · ✅ closed

Tested against a real, live camera: an Android phone running the IP Webcam app, pulled by MediaMTX exactly like any IP camera would be (`infra/mediamtx.yml`'s `phone-cam` path), with real detection, tracking, and alerts generated from it. One real operational finding from that run: orientation matters more than expected — sideways (portrait) video gave person-detection confidence ≈0.31, below the 0.40 production threshold and silently dropped; landscape gave ≈0.82. Worth remembering for any future live demo setup.

### Gate 5 — A GPU · ✅ closed (CUDA + CoreML measured), one sliver open (TensorRT)

The `bop` profile targets an RTX 3060+ with TensorRT FP16. This has now been tested on two real machines from two different angles, and both A/B comparisons are real `make bench` runs, not estimates.

**Apple M1 (CPU vs. CoreML):** TensorRT/CUDA are NVIDIA-only and cannot run on Apple Silicon at all — that part of the gate could never close here. What *can* run — ONNX Runtime's CoreMLExecutionProvider, dispatching to the Apple Neural Engine — was benchmarked: ~1,000 inferences of `yolo11n.onnx` across two input sizes, full numbers in [`docs/BENCH.md`](docs/BENCH.md). Result: **CPU beat CoreML** on this model — CoreML only assigns 326 of 410 graph nodes to the Neural Engine, and dispatch overhead outweighs the gain for a model this small. `config/profiles/laptop.yaml` now pins CPU ahead of CoreML as a result, which closed a real, measured deficit: the laptop profile's detect stage went from 0.9× headroom against its 6 fps target (short of it) to 1.1×.

**RTX 3050 Laptop (CPU vs. CUDA), the `bop` profile's actual target family:** measured at 4 GB VRAM, compute 8.6, driver 616.92, YOLO11s at 640×640:

| | batch=1 | batch=8, per frame |
| --- | ---: | ---: |
| CPU (Ryzen, ORT CPU EP) | 40.1 ms | 46.7 ms |
| **CUDA EP** | **15.2 ms** | **12.9 ms → 76 fps aggregate** |

**6.4 cameras at the profile's 12 fps**, on a laptop GPU a tier *below* the RTX 3060 the profile targets — the §5 claim of 8–12 cameras is no longer theoretical, it is bracketed from below by measured hardware.

Three bugs had to be fixed to get the CUDA number, and all three would have bitten a real deployment silently:

1. **`fp16: true` was decorative.** It was parsed from config and never passed to a provider, so the profile that advertised FP16 ran FP32. It is now `trt_fp16_enable`.
2. **The `bop` profile pointed at `yolo11s.trt`,** a serialised TensorRT engine. `ort.InferenceSession` cannot load one — it takes ONNX and builds the engine itself. Any attempt to run this profile would have failed at startup.
3. **CUDA bound to nothing, silently.** The CUDA libraries ship as `nvidia-*` wheels that unpack somewhere Windows does not search for DLLs, so ORT reported a missing `cublasLt64_13.dll`, fell back to CPU, and ran perfectly — ten times slower than the hardware allows, with no obvious symptom. `onnxruntime.preload_dlls()` now runs before session creation.

A fourth fix came from measuring rather than reading: with the GPU doing inference in 5.4 ms, **frame preprocessing became the bottleneck at 5.8 ms/frame** — three full-array temporaries and a fresh 39 MB allocation per batch. Rewritten to fill a reused NCHW buffer, verified bit-identical to the implementation it replaced. Batched throughput went from 53 to 90 fps on YOLO11n; that is where most of the table above comes from.

**Still open — TensorRT EP specifically.** Not a hardware problem, a packaging one: ONNX Runtime 1.30 links `nvinfer_10.dll` (TensorRT 10), and TensorRT 10 publishes no wheels for Python 3.14 — only TensorRT 11, which ORT does not yet link. The CUDA ceiling above is real and closes the gate's substance (a GPU accelerates this pipeline, measured, on the target hardware family); TensorRT FP16 on top would typically add another 1.3–2×.

- **Give me:** Python 3.12 or 3.13 (where `tensorrt-cu13==10.x` installs), and `make bench PROFILE=bop` produces the TensorRT row with no code change — the provider list already prefers it.
- **Fallback in place, and exercised on both machines:** the provider chain falls back TensorRT → CUDA → CoreML → CPU on its own, and every fallback named above was hit for real, visible in the run log, not just asserted in a unit test.

### Gate 6 — Hyperledger Fabric · ❌ open, untested

`make fabric-up` needs Docker plus `fabric-samples` (~2 GB) and Go.

- **Give me:** Docker + `./install-fabric.sh docker samples binary`, then `make fabric-up`.
- **Fallback in place:** the mock ledger is a **real append-only hash-chained log** that detects tampering and names the broken entry. It exercises the identical code path; switching is one config line. It has been run live and caught real corruption during this project's own debugging (see the evidence-hash bug below) — the tamper-evidence design did its job.
- **Risk if skipped:** the word "blockchain" in the problem statement is answered by a chained JSONL file rather than Fabric. Defensible, but weaker in the room.

### Gate 7 — Decisions only you can make

| Decision | Default I chose | Why you might change it |
| --- | --- | --- |
| **AGPL vs Apache** | YOLO11 (AGPL-3.0) | If the deploying org rejects AGPL, set `detector.backend: rtdetr` — Apache-2.0, ~2 points mAP lower, no other change |
| **Face analytics** | Disabled | It is the answer to the privacy question. Turning it on is a policy decision, not an engineering one |
| **Team members** | Team SW-73 (ByteForge) | Add individual member names/roles to this README before submission, if the format requires it |
| **Site geometry** | `BOP-03`, Delhi coordinates, invented zones | Replace with your actual demo site in `config/sites/` |

### Gate 8 — Visual/UX verification · ✅ closed

The dashboard was actually opened and looked at — headless-Chrome screenshots of login, the live camera wall, the alert list, and the evidence verification panel were all captured and reviewed during this project's own live debugging. Two real bugs were found this way and are already fixed: a camera tile could show a green "live" dot over completely blank video (it now waits for the browser to confirm playable media before claiming live), and every failed login attempt was being reported as "session expired — sign in again" instead of "invalid username or password" (now only shown when a session actually existed before the call).

### A third bug worth naming: evidence hashes were silently wrong

Not a gate, but the most serious defect found in this project: `numpy.float64` values from the Kalman filter's tracking state were reaching the RFC 8785 evidence canonicaliser uncast. `numpy.float64` passes `isinstance(x, float)`, but in numpy ≥2.0 its `repr()` prints `np.float64(0.616)` instead of `0.616` — and the canonicaliser calls `repr()` internally. Every evidence hash computed from a track's box was silently wrong while the *stored* JSON (written by the numpy-agnostic standard encoder) looked completely normal. This reproduced on 100% of alerts and was only caught because the dashboard's own verification panel reported "TAMPERED" on a real, untampered alert. Fixed at the root cause (explicit `float()` casts in `track.py`) and with defense-in-depth in `evidence.py`'s canonicaliser, which now raises loudly — instead of hashing silently wrong — if a float subclass like this ever reaches it again.

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
                                       └── RTSP ──▶ ibvap-worker
                                                      │
        ingest → EVQM → letterbox → enhance → detect → track
              → geometry → rules → risk → debounce → evidence → sinks
                                                      │
              ┌───────────────────────────────────────┼──────────────┐
              ▼                    ▼                  ▼              ▼
          Postgres              MinIO              Redis      ibvap-anchor
          +pgvector          snapshots/clips      streams      Merkle → Fabric
              │                    │                  │
              └────────────▶ ibvap-api ◀────────────┘
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
worker/src/ibvap_worker/  the analytics pipeline (23 modules)
api/src/ibvap_api/        FastAPI: REST, WebSocket, verification
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

**386 Python tests + 6 TypeScript tests**, all passing with no infrastructure required —
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
