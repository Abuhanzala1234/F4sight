# IBVAP — BUILD SPEC

> **Authoritative.** `CLAUDE.md` defers to this document. Interface contracts in §7 are
> frozen; changing one requires an explicit note and a spec edit in the same commit.

| | |
| --- | --- |
| Problem statement | SIH 2026 · PS 26187 · *AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure* |
| Theme | Blockchain & Cybersecurity (Software) |
| Team | SW-73 — ByteForge |
| Codename | IBVAP (*Detection, Recognition & Intelligent Surveillance for High-Threat Infrastructure — Border Out Post*) |
| Spec version | 1.0.0 |

---

## 1. Purpose and scope

### 1.1 The problem, restated

Border Out Posts (BOPs), checkpoints and border roads already have IP CCTV. Those cameras
do two things: show live video, and record it. Everything else is a human sitting in a
room. That human gets tired, and the failure mode of a tired human is silent — nobody
knows which frames were missed.

Replacing the cameras is not on the table. Specialised edge-analytics hardware is
expensive per-camera and procurement-heavy. So the system has to be software that bolts
onto **the RTSP streams that already exist**.

### 1.2 What we build

A self-hosted video analytics platform that:

- ingests existing RTSP/ONVIF camera streams without touching camera configuration;
- runs detection, tracking, intrusion/loitering/direction rules, ANPR and (opt-in) face
  matching on those streams;
- scores every candidate event with an **additive, explainable** risk model;
- raises debounced, evidence-backed alerts to an operator dashboard in real time;
- writes a **tamper-evident audit trail**: every alert's evidence is canonicalised,
  SHA-256 hashed, Merkle-batched and anchored to a Hyperledger Fabric channel;
- runs fully offline on one workstation-class box, from free and open-source parts.

### 1.3 What we do not build

- Camera firmware, PTZ control loops, or anything that writes to a camera.
- Any automated response (barrier, siren, weapon, dispatch). The system recommends; a
  human adjudicates. See P7.
- Biometric enrolment. Faces are matched against an explicitly curated watchlist or not
  at all. See P6.
- Cross-BOP person re-identification. [STRETCH], and a policy question before it is an
  engineering one.

### 1.4 Non-negotiable cost constraint

Every component is free and self-hostable — see §4. The system must complete a cold
start and a full demo with the network cable unplugged. Weight files are vendored
locally under `models/`; nothing calls out at inference time.

---

## 2. Principles

These are referenced by ID throughout the spec and in `CLAUDE.md`.

| ID | Principle | Consequence in code |
| --- | --- | --- |
| **P1** | **Existing infrastructure is the platform.** | RTSP in, HLS out. No camera-side change, no per-camera hardware. Degrade gracefully on 4CIF/H.264 baseline streams from 2014. |
| **P2** | **An alert that cannot be explained must not be raised.** | Every alert carries `reason_codes[]` and a `risk_breakdown[]` whose contributions sum exactly to `risk_score`. |
| **P3** | **False alarms are the enemy, not missed detections.** | When in doubt, raise the threshold. An operator who has stopped trusting the banner has an effective recall of zero. |
| **P4** | **Evidence is for the court, not for the model.** | Snapshots and clips are original, unenhanced frames. Enhancement parameters are stored as metadata beside them. |
| **P5** | **Tamper-evidence is a property of the record, not of the database.** | Canonicalise → hash → Merkle → anchor. Verification must be reproducible by a third party with only the evidence JSON and the ledger. |
| **P6** | **Privacy is a default, not a setting.** | Face analytics ships disabled. Plate text is HMAC'd at rest. Non-matching embeddings are destroyed at track close. |
| **P7** | **The human adjudicates.** | No automated response action. Alerts have exactly one lifecycle: raised → acknowledged → adjudicated (true/false/unclear). |
| **P8** | **Never lose video because AI broke.** | The recording path and the analytics path are separate processes. Analytics crashing must not stop MediaMTX recording. |
| **P9** | **Offline is the normal case.** | A BOP has intermittent connectivity. Ledger anchoring, model downloads and updates are all queue-and-retry, never blocking. |
| **P10** | **Free and auditable beats proprietary and better.** | No paid API. Every model's licence and provenance is recorded in `docs/MODELS.md`. |

---

## 3. Architecture

### 3.1 Process topology

```
 ┌──────────────┐   RTSP    ┌───────────────────────────────────────────┐
 │ Existing IP  │──────────▶│ MediaMTX  (recording + restream)          │
 │ CCTV cameras │           │  · RTSP proxy   · HLS/LL-HLS  · WebRTC    │
 └──────────────┘           │  · segment recorder → ./recordings        │
                            └───────┬───────────────────────┬───────────┘
                        RTSP (local)│                       │ HLS
                                    ▼                       │
 ┌──────────────────────────────────────────────┐           │
 │ ibvap-worker        (one process per host, │           │
 │                        one thread-group/cam) │           │
 │  ingest → EVQM → enhance → detect → track    │           │
 │        → geometry → rules → risk → debounce  │           │
 │        → evidence → sinks                    │           │
 └────┬───────────────┬──────────────┬──────────┘           │
      │ SQL           │ S3           │ pub/sub              │
      ▼               ▼              ▼                      │
 ┌──────────┐  ┌────────────┐  ┌──────────┐                 │
 │ Postgres │  │   MinIO    │  │  Redis   │                 │
 │ +pgvector│  │ snapshots  │  │ streams  │                 │
 └────┬─────┘  │   clips    │  └────┬─────┘                 │
      │        └──────┬─────┘       │                       │
      ▼               ▼             ▼                       │
 ┌──────────────────────────────────────────────┐           │
 │ ibvap-api     (FastAPI, async)             │           │
 │  REST · WebSocket fan-out · evidence verify  │           │
 └────┬─────────────────────────────┬───────────┘           │
      │ REST/WS                     │ gRPC                  │
      ▼                             ▼                       ▼
 ┌──────────────────────┐   ┌────────────────┐   ┌────────────────────┐
 │ React command center │   │ anchor-service │   │  <video> via hls.js │
 │  live wall · alerts  │   │ Merkle batcher │   └────────────────────┘
 │  evidence · verify   │   │  → Fabric      │
 └──────────────────────┘   └────────────────┘
```

Four deployable processes: `mediamtx`, `ibvap-worker`, `ibvap-api`,
`ibvap-anchor`. Plus three stateful services: Postgres, MinIO, Redis. The dashboard is
static files served by the API in production, Vite in dev.

### 3.2 Why the worker is one process, threaded

Per camera we run a thread group, not an asyncio task group and not a subprocess:

- **Not async:** the hot path is CPU/GPU-bound (`cv2` decode, ONNX/TensorRT inference).
  `asyncio` would serialise it behind the GIL with extra ceremony. The heavy calls
  (`cv2.VideoCapture.read`, `ort.InferenceSession.run`, `trt.execute_v2`) all release the
  GIL, so threads genuinely parallelise.
- **Not per-camera processes:** the GPU/accelerator is a single shared resource and we
  batch across cameras in one inference thread. N processes means N model copies in VRAM.

Per camera: one **reader thread** (decode + watchdog) and one **stage thread** (everything
downstream of detection). One **shared inference thread** per host drains a batch queue
across all cameras. §7.14 pins the exact queue topology and backpressure rules.

### 3.3 Data flow for one alert

1. Reader thread decodes frame `F` at `t`, stamps `FrameRef(camera_id, frame_id, ts_utc)`.
2. EVQM samples `F` every `sample_every_n` frames, updates the quality state machine
   (§7.2), possibly switches the processing profile (with hysteresis).
3. The frame is letterboxed to model size, **then** enhanced (§7.3) to produce `F'`.
   That order is deliberate: enhancing at model scale touches roughly a quarter of the
   pixels (measured: full-frame fog dehaze 50 ms vs 6 ms post-resize), and enhancing
   before padding keeps the black bars out of CLAHE's histogram. `F` is retained
   untouched for evidence (P4), and the `FrameTransform` travels with the frame.
4. Detector (§7.4) returns boxes in `F'` space; the pipeline maps them back through
   `FrameTransform` into `F` space before anything else sees them.
5. Tracker (§7.5) associates detections into `Track`s with stable `track_id`.
6. Geometry (§7.6) evaluates each track's foot-point against zone polygons and tripwires.
7. Rules (§7.7) turn geometric facts into `Signal`s. Risk (§7.8) turns signals into an
   additive score with a breakdown.
8. Debouncer (§7.7.4) suppresses repeats; survivors become `AlertCandidate`s.
9. Evidence (§7.11) writes an unenhanced snapshot + pre/post-roll clip to MinIO,
   assembles the evidence document, canonicalises it (RFC 8785) and SHA-256 hashes it.
10. Sinks (§7.13) insert the alert row, publish to Redis, and enqueue a ledger anchor.
11. API fans the alert out over WebSocket; the dashboard banners it.
12. `ibvap-anchor` batches hashes into a Merkle tree every `anchor_interval_s`, writes
    the root to Fabric, and stores each alert's inclusion proof.

### 3.4 Latency budget (glass-to-banner)

| Stage | Budget |
| --- | --- |
| Camera → MediaMTX → worker decode | 250 ms |
| Queue wait + preprocess | 40 ms |
| Inference (batched, 640×640) | 45 ms |
| Track + geometry + rules + risk | 15 ms |
| Evidence snapshot (clip is async) | 60 ms |
| DB insert + Redis publish | 25 ms |
| WebSocket → React render | 65 ms |
| **Total** | **≤ 500 ms p95** |

Clip assembly, ledger anchoring and ANPR OCR are explicitly **off** this path.

---

## 4. The free stack

Every row is free to use, self-hostable, and works offline after first download. §4.3
records what we deliberately did not use and why.

### 4.1 Models

| Role | Model | Licence | Size | Runtime |
| --- | --- | --- | --- | --- |
| Object detection **[DEMO-CRITICAL]** | YOLO11n / YOLO11s (Ultralytics) exported to ONNX | AGPL-3.0 | 10 / 36 MB | ONNX Runtime (CPU/CUDA) or TensorRT |
| Object detection (permissive fallback) | RT-DETR-R18 ONNX | Apache-2.0 | 77 MB | ONNX Runtime |
| Tracking | ByteTrack — reimplemented in-repo, no weights | MIT (algorithm) | 0 | pure Python + SciPy |
| Plate detection | YOLO11n fine-tuned on CCPD + Indian-plate open sets | AGPL-3.0 | 10 MB | ONNX Runtime |
| Plate OCR | PaddleOCR v4 `en_PP-OCRv4_rec` (+ `ch_PP-OCRv4_det`) | Apache-2.0 | 12 MB | ONNX Runtime |
| Plate OCR (fallback) | Tesseract 5, `--psm 7`, plate charset | Apache-2.0 | — | pytesseract |
| Face detection | SCRFD-500M (InsightFace `buffalo_s`) | MIT code / free weights | 3 MB | ONNX Runtime |
| Face embedding | ArcFace w600k-mbf (InsightFace `buffalo_s`) | MIT code / free weights | 13 MB | ONNX Runtime |
| Low-light enhancement | Zero-DCE++ | MIT | 0.03 MB | ONNX Runtime |
| Dehaze | Dark Channel Prior — classical, no weights | — | 0 | OpenCV |
| Night/thermal-ish contrast | CLAHE — classical | — | 0 | OpenCV |

`make models` downloads and exports all of the above into `models/` and writes
`models/MANIFEST.json` with a SHA-256 per file. The worker refuses to start if a manifest
hash mismatches (supply-chain hygiene, and it catches half-downloaded files).

**Licence note.** Ultralytics YOLO is AGPL-3.0: fine for a hackathon and for a government
deployment that does not redistribute a modified closed binary. `detector.backend=rtdetr`
switches to the Apache-2.0 path with no other code change (§7.4). Both are free.

### 4.2 Infrastructure

| Role | Choice | Licence | Why |
| --- | --- | --- | --- |
| Relational store | PostgreSQL 16 | PostgreSQL Licence | Free, boring, correct. |
| Vector search | pgvector 0.7 | PostgreSQL Licence | Face embeddings without a second datastore. |
| Time-series rollups | TimescaleDB community *(optional)* | Apache-2.0 | Only for `make bench` charts. Plain Postgres if absent. |
| Object storage | MinIO | AGPL-3.0 | S3 API offline; snapshots and clips. |
| Cache / event bus | Redis 7.2 (or Valkey 8, drop-in) | BSD-3 (Valkey) | Redis Streams for worker→API fan-out. |
| Media server | MediaMTX | MIT | RTSP proxy, **recording**, and RTSP→HLS/WebRTC. Solves blocker #4. |
| Ledger | Hyperledger Fabric 2.5 test network | Apache-2.0 | The PS theme. Mock backend is the default (§7.12). |
| API | FastAPI + Uvicorn | MIT | Async, Pydantic v2 native. |
| ORM / migrations | SQLAlchemy 2 + Alembic | MIT | |
| Dashboard | React 18 + Vite + TypeScript + Tailwind | MIT | |
| Video in browser | hls.js | Apache-2.0 | |
| Inference | ONNX Runtime (+ TensorRT EP when a GPU exists) | MIT / Apache-2.0 | One code path, three backends. |
| Tests | pytest, hypothesis, vitest | MIT | |

### 4.3 Rejected, and why

| Rejected | Reason |
| --- | --- |
| DeepStream / Triton | Free-as-in-beer but NVIDIA-only and heavy; breaks the "runs on the evaluator's laptop" requirement. |
| Any hosted vision or OCR API | Violates P9 and P10, and the network at a BOP is not a given. |
| Ethereum / any public chain | Gas costs money, and border evidence does not belong on a public ledger. Fabric is permissioned and free. |
| Milvus / Qdrant | A whole extra service for ≤10k watchlist vectors. pgvector is enough. |
| Kafka | Redis Streams covers one-host fan-out. Kafka is an ops tax we cannot pay in a hackathon. |

---

## 5. Deployment profiles

The same artefacts run in three shapes. `config/profiles/*.yaml` selects one.

| Profile | Target | Cameras | Detector | Notes |
| --- | --- | --- | --- | --- |
| `laptop` | Evaluator's machine, CPU only | 1–2 @ 6 fps | YOLO11n ONNX CPU, 480×480 | The default. `make demo` uses this. Must work on macOS/arm64. |
| `bop` | One workstation w/ RTX 3060+ | 8–12 @ 12 fps | YOLO11s TensorRT FP16, 640×640 | The realistic field deployment. |
| `edge` | Jetson Orin Nano | 4 @ 10 fps | YOLO11n TensorRT INT8, 512×512 | [STRETCH] |

**Frame-rate honesty.** We do not process every frame. Analytics runs at
`camera.analytics_fps` (default 6) by dropping frames at the reader; recording stays at
full camera rate in MediaMTX (P8). A person walking at 1.4 m/s moves 23 cm between
analytics frames at 6 fps — comfortably inside tracker association distance.

---

## 6. Data model

PostgreSQL. UUIDv7 primary keys everywhere (`uuid7()` in `infra/postgres/init.sql`) —
time-sortable, which matters for evidence ordering. All timestamps are `TIMESTAMPTZ` in
UTC; the dashboard localises. Migrations live in `api/src/ibvap_api/migrations/`.

### 6.1 Entity relationships

```
site ──< camera ──< zone
                └─< stream_health
site ──< user_site_role >── user
camera ──< track ──< detection_sample
camera ──< event ──< alert ──< evidence_item
                        └──< alert_action
alert ──1 ledger_anchor_entry >── ledger_anchor_batch
watchlist_person ──< watchlist_face_vector
watchlist_vehicle
audit_log
config_version
```

### 6.2 Core tables

**`site`** — a BOP, checkpoint or road segment.
`id, code (unique, e.g. BOP-03), name, sector, lat, lon, timezone, created_at`

**`camera`** — one physical CCTV camera.
`id, site_id→site, code (unique per site), name, rtsp_url (encrypted at rest),
mediamtx_path, resolution_w, resolution_h, native_fps, analytics_fps, is_recording_only,
profile ('laptop'|'bop'|'edge'), lens ('fixed'|'ptz'), calibration jsonb, enabled,
created_at`

- `is_recording_only = true` → MediaMTX records it, the worker ignores it. P8's escape
  hatch and the way we survive a camera the model hates.
- `calibration jsonb`: optional `{homography: [[9 floats]], px_per_m_at_y: [[y, m]...]}`
  used by §7.6 for real-world speed. Absent ⇒ speed reported in px/s and speed rules are
  disabled rather than guessed.

**`zone`** — a polygon or tripwire drawn on a camera's image plane.
`id, camera_id→camera, name, kind ('area'|'tripwire'|'mask'), polygon jsonb (normalised
[[x,y]…] in 0..1), direction ('in'|'out'|'both'|null), classes text[], schedule jsonb,
severity_base smallint, enabled, created_at`

- Normalised coordinates so a resolution change does not invalidate every zone.
- `kind='mask'` is a **negative** zone: detections inside it are dropped before rules
  (the flapping tree, the road with civilian traffic).
- `schedule jsonb`: `{"tz":"Asia/Kolkata","windows":[{"days":[0..6],"from":"18:00","to":"06:00"}]}`.
  Null ⇒ always active.

**`track`** — one tracked object's life.
`id, camera_id→camera, track_id int (per-camera, recycled), cls, first_seen_at,
last_seen_at, frame_count, max_conf, path jsonb (downsampled [[t,x,y]…]),
attributes jsonb, closed_at`

**`event`** — a rule firing, pre-debounce. Cheap, high-volume, retained 7 days.
`id, camera_id, track_id→track, zone_id→zone, kind, ts_utc, payload jsonb`

**`alert`** — a debounced, scored, operator-visible thing. **[DEMO-CRITICAL]**
`id, site_id, camera_id, track_id, primary_event_id, kind, severity
('info'|'low'|'medium'|'high'|'critical'), risk_score numeric(5,2),
risk_breakdown jsonb, reason_codes text[], ts_utc, window_start, window_end,
status ('raised'|'acknowledged'|'adjudicated'), adjudication
('true_positive'|'false_positive'|'unclear'|null), evidence_hash char(64),
evidence_doc jsonb, ledger_status ('pending'|'anchored'|'failed'), created_at`

- `risk_breakdown`: `[{"code":"ZONE_INTRUSION","weight":40.0,"detail":{...}}, …]`.
  **Invariant:** `sum(weight) == risk_score`, checked in code and by a DB `CHECK`-backed
  trigger. P2.
- `evidence_doc` is the exact object that was hashed, minus `evidence_hash` and `ledger`.
  Storing it makes verification a pure function of one row.

**`evidence_item`** — a file in MinIO.
`id, alert_id→alert, kind ('snapshot'|'clip'|'crop'|'plate_crop'), bucket, object_key,
sha256, bytes, width, height, captured_at, enhanced boolean default false,
enhancement_params jsonb`

- `enhanced` must be `false` for `snapshot` and `clip` (P4); a DB constraint enforces it.

**`ledger_anchor_batch`** / **`ledger_anchor_entry`**
`batch: id, merkle_root char(64), leaf_count, backend ('mock'|'fabric'), tx_id,
block_number, anchored_at, status`
`entry: id, batch_id→batch, alert_id→alert, leaf_hash char(64), leaf_index,
proof jsonb (sibling path), created_at`

**`watchlist_person`** / **`watchlist_face_vector`** — opt-in, P6.
`person: id, ref_code, display_name, category, added_by→user, expires_at, active, notes`
`vector: id, person_id, embedding vector(512), source_image_key, created_at`

**`watchlist_vehicle`**
`id, plate_hmac char(64) unique, plate_ciphertext bytea, region, category, added_by,
expires_at, active`

- We store the HMAC for lookup and an authenticated-encrypted blob for the rare
  legitimate reveal. Never plaintext. P6.

**`user`**, **`user_site_role`** — roles `viewer | operator | investigator | admin`.
**`audit_log`** — `id, actor_id, action, target_type, target_id, ts_utc, ip, detail jsonb`.
Append-only; every evidence view and every plate reveal writes a row.
**`stream_health`** — `camera_id, ts_utc, state, fps_in, fps_analytics, reconnects,
last_error` — 60 s rollups powering the dashboard's health strip.
**`config_version`** — hash + snapshot of `config/` at worker start, referenced by
`alert.evidence_doc.config_version`. You must be able to say *which thresholds produced
this alert*.

### 6.3 Retention

| Data | Retention | Mechanism |
| --- | --- | --- |
| MediaMTX recordings | 7 days | MediaMTX `deleteAfter` |
| `event` | 7 days | nightly `DELETE` |
| `track` | 7 days | nightly `DELETE` |
| Alerts + evidence + ledger | 1 year | never auto-deleted in the demo |
| Non-matching face embeddings | **0 — destroyed at track close** | in-memory only, never written (P6) |
| `audit_log` | 1 year | append-only |

---

## 7. Module contracts — **FROZEN**

Every signature below is authoritative. All live in `worker/src/ibvap_worker/`.
Shared value types are in `types.py`; all are `@dataclass(frozen=True, slots=True)`.

```python
# types.py — the vocabulary. Frozen values, no behaviour beyond derived properties.

Point   = tuple[float, float]          # pixels in ORIGINAL frame space unless stated
BoxXYXY = tuple[float, float, float, float]

@dataclass(frozen=True, slots=True)
class FrameTransform:
    """Maps model-input space back to original-frame space. Never optional. (Invariant)"""
    scale_x: float; scale_y: float
    pad_x: float;   pad_y: float
    crop_x: float = 0.0; crop_y: float = 0.0   # for tiled inference
    def to_original(self, b: BoxXYXY) -> BoxXYXY: ...
    @staticmethod
    def letterbox(src_wh, dst_wh) -> "FrameTransform": ...

@dataclass(frozen=True, slots=True)
class Frame:
    camera_id: str
    frame_id: int                  # monotonic per camera, never reused
    ts_utc: datetime
    image: np.ndarray              # BGR uint8, ORIGINAL, never mutated (P4)
    width: int; height: int
    seq_gap: int = 0               # frames dropped before this one

@dataclass(frozen=True, slots=True)
class Detection:
    cls: str                       # 'person'|'vehicle'|'animal'|'bag'|...
    conf: float
    box: BoxXYXY                   # ORIGINAL frame coords
    cls_id: int

@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    cls: str
    box: BoxXYXY
    conf: float
    age_frames: int                # frames since track birth
    hits: int                      # frames with a matched detection
    time_since_update: int
    first_seen: datetime
    last_seen: datetime
    history: tuple[Point, ...]     # foot-points, oldest→newest, capped
    @property
    def foot_point(self) -> Point: ...     # (cx, y2) — where the object meets the ground
    @property
    def is_confirmed(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class Signal:
    code: str                      # ZONE_INTRUSION, TRIPWIRE_CROSS, LOITER, ...
    weight: float                  # risk points, may be negative
    detail: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class RiskResult:
    score: float                   # clamped to [0, 100]
    breakdown: tuple[Signal, ...]
    severity: str
    # INVARIANT: round(sum(s.weight for s in breakdown), 2) == score, pre-clamp.
```

### 7.1 Ingest — `ingest.py` **[DEMO-CRITICAL]**

Blocker #2 lives here. `cv2.VideoCapture.read()` on a dropped RTSP stream returns
`False` forever, or worse, blocks inside FFmpeg for 30+ seconds. Neither is detectable
from the return value alone, so the watchdog is a separate thread that watches a
timestamp.

```python
class StreamState(StrEnum):
    CONNECTING = "connecting"; LIVE = "live"; STALLED = "stalled"
    RECONNECTING = "reconnecting"; FAILED = "failed"

@dataclass(frozen=True, slots=True)
class IngestConfig:
    rtsp_url: str
    analytics_fps: float = 6.0
    read_timeout_s: float = 5.0        # no frame for this long ⇒ STALLED
    reconnect_backoff_s: tuple[float, ...] = (1, 2, 4, 8, 15, 30)
    max_consecutive_failures: int = 0  # 0 = retry forever
    transport: str = "tcp"             # RTSP over TCP; UDP loses frames silently
    buffer_size: int = 1               # CAP_PROP_BUFFERSIZE — we want latest, not queued

class RtspReader:
    """One reader thread + one watchdog thread per camera."""
    def __init__(self, camera_id: str, cfg: IngestConfig,
                 on_frame: Callable[[Frame], None],
                 on_state: Callable[[str, StreamState, str | None], None]) -> None: ...
    def start(self) -> None: ...
    def stop(self, timeout_s: float = 5.0) -> None: ...
    @property
    def state(self) -> StreamState: ...
    @property
    def stats(self) -> IngestStats: ...   # fps_in, fps_emitted, reconnects, last_error
```

Mandatory behaviour:

1. Open with `cv2.CAP_FFMPEG` and `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|stimeout;5000000`.
2. Set `CAP_PROP_BUFFERSIZE=1`. We want the newest frame; a backed-up buffer is latency
   we can never recover.
3. Reader loop updates `self._last_frame_mono = time.monotonic()` **before** invoking
   `on_frame`.
4. Watchdog thread: every 1 s, if `monotonic() - _last_frame_mono > read_timeout_s`,
   transition to `STALLED`, **release the capture from the watchdog thread** (this is what
   unblocks a wedged `read()`), and signal the reader to reconnect.
5. Reconnect with the backoff tuple, saturating at the last value. Every attempt logs at
   WARNING with camera id, attempt number and the last error.
6. Frame dropping to hit `analytics_fps` happens **here**, by timestamp, not by
   `frame_id % n` — variable-rate sources exist.
7. `on_frame` must never raise into the reader. Wrap it; log with context; drop the frame.
   A slow consumer must not stall decode.
8. On `stop()`, threads join within `timeout_s` or are abandoned with an ERROR log.

`FileReader(path, loop=True)` implements the same interface for fixture MP4s so tests
never need a network. §16 Phase 2 acceptance uses MediaMTX-served fixtures anyway.

### 7.2 EVQM — `evqm.py` (Environmental & Video Quality Monitor)

Blocker #5. Measures what the scene is doing to the image, and picks a processing
profile. **Hysteresis is mandatory** — without it a headlight sweep flips the profile
every second and the detector's thresholds oscillate.

```python
@dataclass(frozen=True, slots=True)
class QualityMetrics:
    brightness: float      # 0..1, mean V of HSV
    contrast: float        # 0..1, normalised std of luma
    blur: float            # 0..1, 1 = sharp; variance of Laplacian, normalised
    fog: float             # 0..1, 1 = heavy fog; dark channel prior
    noise: float           # 0..1, high-freq energy after median subtraction
    motion: float          # 0..1, fraction of pixels changed vs previous sample
    def profile_vote(self, cfg) -> str: ...   # pure

class EVQM:
    def __init__(self, cfg: EVQMConfig) -> None: ...
    def observe(self, frame: Frame) -> QualityMetrics | None:  # None on non-sampled frames
        ...
    @property
    def profile(self) -> str:      # 'day'|'lowlight'|'night'|'fog'|'degraded'
        ...
    @property
    def metrics(self) -> QualityMetrics | None: ...
```

- Sampled every `sample_every_n` frames (default 15) on a 320-px-wide downscale. EVQM must
  cost < 3 ms; it is not allowed to be interesting.
- **Hysteresis:** a candidate profile must win `enter_samples=3` consecutive votes to be
  entered, and the current profile must lose `exit_samples=5` consecutive votes to be
  left. Entering is faster than leaving, deliberately.
- All thresholds in `config/evqm.yaml`. Never in source.
- `fog` uses the dark channel prior: `min` over a 15-px window of per-pixel channel minima;
  a high dark-channel mean means haze.
- Profile changes emit a structured log line and a `stream_health` row. The operator can
  see *why* the system changed its mind.

### 7.3 Enhancement — `enhance.py` **hot path**

```python
@dataclass(frozen=True, slots=True)
class EnhancementResult:
    image: np.ndarray                 # the image the MODEL sees
    params: Mapping[str, Any]         # recorded into evidence metadata (P4)
    applied: tuple[str, ...]

def enhance_for_model(image, profile: str, cfg) -> EnhancementResult: ...
```

**P4 is absolute here.** This function returns a new array; the caller keeps the original
`Frame.image` untouched for evidence. Per profile: `day` → none (identity, zero copy);
`lowlight` → CLAHE on L of LAB; `night` → CLAHE + Zero-DCE++; `fog` → dark-channel
dehaze + mild unsharp; `degraded` → denoise + CLAHE. Budget 8 ms at 640×640 — which is why `pipeline.py`
calls this **after** the letterbox resize, not on the full frame. Dehazing computes its
transmission map at 1/4 scale and upsamples it; the field is smooth, so this is visually
free and it is the difference between 50 ms and 6 ms.

### 7.4 Detection — `detect/` **[DEMO-CRITICAL]** **hot path**

```python
class Detector(Protocol):
    def infer(self, images: Sequence[np.ndarray]) -> list[list[RawDetection]]: ...
    def warmup(self, n: int = 10) -> float: ...      # returns seconds spent
    @property
    def input_size(self) -> tuple[int, int]: ...
    @property
    def backend(self) -> str: ...

def build_detector(cfg: DetectorConfig) -> Detector: ...   # factory, reads config only
```

Backends: `onnx` (CPU/CUDA/CoreML EP), `tensorrt` (FP16/INT8 engine), `mock`
(deterministic, for tests — returns scripted boxes, never touches a GPU).

- **Blocker #3:** `build_detector` calls `warmup(10)` before the pipeline starts and logs
  the time. TensorRT first inference is ~2 s; ORT graph optimisation is ~400 ms. The
  worker does not report `ready` until warmup completes.
- Letterbox preprocessing produces the `FrameTransform`. The detector returns boxes in
  **model space**; `pipeline.py` maps them to original space immediately. No other module
  ever sees model-space coordinates. (Invariant.)
- Class-agnostic NMS at `iou=0.45`, then per-class confidence thresholds from
  `config/detector.yaml` — `person` and `vehicle` deserve different thresholds and P3 says
  err high.
- COCO classes are remapped to our taxonomy by `CLASS_MAP` in config:
  `person→person`; `car|truck|bus|motorcycle|bicycle→vehicle` (with `vehicle_type` as an
  attribute); `dog|cow|horse|sheep→animal`; `backpack|handbag|suitcase→bag`.
  Everything else is dropped before it reaches the tracker.
- Batching: the shared inference thread drains up to `max_batch` (default 8) frames from
  the queue, or fires after `max_wait_ms` (default 15). Never wait for a full batch.

### 7.5 Tracking — `track.py` **hot path**

ByteTrack, reimplemented (MIT algorithm, ~250 lines). The second association pass over
*low*-confidence detections is the whole point: it is what keeps a track alive through
the occlusion behind a fence post, which is exactly where border intrusions happen.

```python
@dataclass(frozen=True, slots=True)
class TrackerConfig:
    high_thresh: float = 0.50
    low_thresh: float = 0.10
    match_thresh: float = 0.80       # 1 - IoU distance for pass 1
    match_thresh_low: float = 0.50
    max_age: int = 30                # frames to keep a lost track
    min_hits: int = 3                # before is_confirmed
    history_len: int = 120

class ByteTracker:
    def __init__(self, cfg: TrackerConfig) -> None: ...
    def update(self, dets: Sequence[Detection], ts: datetime) -> list[Track]: ...
    def close_expired(self) -> list[Track]: ...   # returns tracks that just died
```

- Kalman filter (constant-velocity, 8-state `[x, y, a, h, vx, vy, va, vh]`) for prediction;
  Hungarian assignment on IoU distance via `scipy.optimize.linear_sum_assignment`.
- `update()` is pure with respect to time: it never calls `now()`. `ts` is passed in.
  This is what makes the tracker testable at 1000× speed.
- `close_expired()` returning dead tracks is the hook the face module uses to **destroy
  non-matching embeddings** (P6). It is not optional plumbing.

### 7.6 Geometry — `geometry.py` (pure functions, property-tested)

```python
def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool: ...    # ray casting, edge-inclusive
def segments_intersect(p1, p2, q1, q2) -> bool: ...                    # orientation test
def crossing_direction(prev: Point, cur: Point, wire: tuple[Point, Point]) -> str | None:
    """'in' | 'out' | None — sign of the cross product of the wire normal."""
def denormalise(poly: Sequence[Point], w: int, h: int) -> list[Point]: ...
def polygon_area(poly) -> float: ...
def dwell_seconds(track: Track, poly, fps: float) -> float: ...
def speed_px_per_s(track: Track, fps: float) -> float: ...
def speed_m_per_s(track: Track, fps: float, calib: Calibration | None) -> float | None:
    """None when uncalibrated. NEVER guess a scale — a wrong speed is a wrong alert."""
```

Property tests (`hypothesis`) must cover, at minimum:
- a point strictly inside a convex polygon is always inside under translation/rotation;
- a point on an edge is deterministic (we define edge-inclusive) and never flaps;
- `segments_intersect` is symmetric in both argument pairs;
- crossing the same wire in opposite directions yields opposite labels;
- degenerate input (zero-length wire, polygon with < 3 points, NaN) raises `ValueError`,
  never returns a silent wrong answer.

### 7.7 Rules — `rules.py` **[DEMO-CRITICAL]**

```python
class Rule(Protocol):
    code: str
    def evaluate(self, track: Track, ctx: RuleContext) -> Signal | None: ...

@dataclass(frozen=True, slots=True)
class RuleContext:
    camera: CameraRuntime
    zones: Sequence[ZoneRuntime]
    now: datetime
    profile: str                 # EVQM profile — rules may soften at night
    prev_state: TrackZoneState   # where this track was last frame
```

Shipped rules, all config-driven from `config/rules.yaml`:

| Code | Fires when |
| --- | --- |
| `ZONE_INTRUSION` | Confirmed track's foot-point enters an `area` zone, class in `zone.classes`, zone in schedule. |
| `TRIPWIRE_CROSS` | Foot-point segment crosses a `tripwire` in the configured direction. |
| `LOITER` | Dwell in zone ≥ `loiter_seconds` (default 30). |
| `NIGHT_MOVEMENT` | Any confirmed person track while EVQM profile ∈ {night, lowlight} and local time in the night window. |
| `PERIMETER_APPROACH` | Track's distance to a `tripwire` closes below `approach_px` on a monotonic trend. |
| `UNAUTHORISED_VEHICLE` | Vehicle track in a zone whose `classes` excludes vehicles, or plate not on the allow list. |
| `WATCHLIST_FACE` | Face match ≥ `face_match_threshold` (only if faces enabled). |
| `WATCHLIST_PLATE` | Plate HMAC hits `watchlist_vehicle`. |
| `CROWD_FORMING` | ≥ `crowd_min` person tracks inside one zone simultaneously. |
| `ABANDONED_OBJECT` | `bag` track stationary ≥ `abandoned_seconds` with no person track within `abandon_radius_px`. [STRETCH] |
| `CAMERA_TAMPER` | EVQM reports a step change in brightness/blur inconsistent with time of day. |

**7.7.1 `min_track_age`.** No rule fires on a track with `hits < min_hits` (3) and
`age_frames < min_track_age` (default 8). This alone removes most detector flicker. P3.

**7.7.2 Mask zones** are subtracted before any rule runs. A detection inside a `mask`
does not exist.

**7.7.3 Schedules.** A zone outside its schedule window contributes nothing. Evaluated in
the site's timezone, not the server's.

**7.7.4 Debounce — blocker #1.** The single most demo-critical piece of logic here.

```python
@dataclass(frozen=True, slots=True)
class DebounceConfig:
    cooldown_s: float = 45.0        # same (track, zone, rule) ⇒ suppressed
    escalate_after_s: float = 120.0 # unless still firing this long ⇒ re-raise, escalated
    correlate_window_s: float = 8.0 # merge signals across rules into ONE alert
    max_alerts_per_camera_per_min: int = 6   # hard ceiling, drops with a WARNING log

class Debouncer:
    def submit(self, cand: AlertCandidate) -> AlertDecision: ...
    # AlertDecision ∈ {EMIT, SUPPRESS, MERGE(into_alert_id), ESCALATE}
```

Key: the debounce key is `(camera_id, track_id, zone_id, rule_code)` — **not** just
`rule_code`. A second, different intruder must not be suppressed by the first. Correlation
across rules within `correlate_window_s` for the *same track* produces one alert carrying
several reason codes — which is also what makes the risk breakdown rich instead of noisy.

### 7.8 Risk — `risk.py` **[DEMO-CRITICAL]** (pure, property-tested)

Additive and explainable by construction. P2.

```python
def score(signals: Sequence[Signal], ctx: RiskContext) -> RiskResult: ...
```

Base weights (`config/risk.yaml`, shown with defaults):

| Signal | Weight |
| --- | --- |
| `ZONE_INTRUSION` | +40 × `zone.severity_base/3` |
| `TRIPWIRE_CROSS` (inbound) | +45 |
| `TRIPWIRE_CROSS` (outbound) | +20 |
| `LOITER` | +15, +5 per extra 30 s, capped +30 |
| `NIGHT_MOVEMENT` | +20 |
| `WATCHLIST_FACE` | +35 |
| `WATCHLIST_PLATE` | +30 |
| `CROWD_FORMING` | +25 |
| `UNAUTHORISED_VEHICLE` | +25 |
| `CAMERA_TAMPER` | +30 |
| `LOW_CONFIDENCE` (track max_conf < 0.6) | **−10** |
| `DEGRADED_INPUT` (EVQM = degraded/fog) | **−8** |
| `SHORT_TRACK` (age < 15 frames) | **−12** |
| `KNOWN_PATROL_WINDOW` | **−15** |

The negative signals are the P3 mechanism: they are how the system says *"I saw
something, but the conditions were bad and I am telling you so"* rather than crying wolf
at full volume.

Rules the implementation must satisfy (and `hypothesis` must check):
1. `round(sum(weights), 2) == score` **before** clamping; the clamp is reported as a
   synthetic `CLAMP` signal so the sum still holds after. (Invariant.)
2. `score ∈ [0, 100]` always.
3. Adding a signal with weight ≥ 0 never decreases the score. Monotonic.
4. Severity thresholds: `<20 info`, `<40 low`, `<60 medium`, `<80 high`, `≥80 critical`.
5. `score(())` is exactly `RiskResult(0.0, (), 'info')` — the empty case is not special-cased
   anywhere else.

### 7.9 ANPR — `anpr.py`

Two-stage, off the hot path (runs in a worker pool fed by track crops).

```python
@dataclass(frozen=True, slots=True)
class PlateRead:
    text: str; conf: float; box: BoxXYXY; region: str | None
    char_confs: tuple[float, ...]; frames_agreed: int

def read_plate(crops: Sequence[np.ndarray], cfg) -> PlateRead | None: ...
def validate_indian_plate(text: str) -> tuple[bool, str]:
    """Returns (is_valid, normalised). Handles LLNNLLNNNN, BH-series, and
    O/0, I/1, S/5, B/8, Z/2 confusions, which are 80% of real OCR error."""
def plate_hmac(text: str, key: bytes) -> str: ...    # HMAC-SHA256, hex
```

- **Multi-frame voting is mandatory:** a plate is only accepted after the same normalised
  text wins `min_frames_agreed` (default 3) reads across the track. Single-frame OCR on
  CCTV is not trustworthy and P3 applies.
- `validate_indian_plate` gets `hypothesis` tests: any valid plate survives a
  normalise→validate round trip; any string with a disallowed structure is rejected; the
  confusion-correction map is idempotent.
- **Storage (P6):** `plate_hmac` is what goes into the DB and into any non-fired-alert
  record. Plaintext appears only inside `evidence_doc` of an alert that actually fired, and
  reading it writes an `audit_log` row.

### 7.10 Faces — `faces.py` — **disabled by default (P6)**

```python
def detect_faces(image, cfg) -> list[FaceBox]: ...
def embed(image, faces, cfg) -> list[np.ndarray]: ...     # 512-d, L2-normalised
def match(emb, index: WatchlistIndex, threshold: float) -> FaceMatch | None: ...
```

- `config/faces.yaml` ships `enabled: false`. The module refuses to load weights unless
  enabled; the dashboard shows the feature as off with a one-line rationale.
- Cosine similarity against pgvector, `threshold` default **0.55** (deliberately strict).
- **Embeddings for non-matching tracks are held in memory only and destroyed in the
  `close_expired()` callback.** There is no code path that writes an embedding to the DB
  except an explicit admin enrolment through the API, which writes an `audit_log` row.
  (Invariant; test `test_faces_no_silent_enrolment` asserts zero DB writes on a
  non-matching track's full lifecycle.)

### 7.11 Evidence — `evidence.py` **[DEMO-CRITICAL]**

Blocker #7. Get canonicalisation wrong and verification fails silently months later.

```python
def canonicalise(doc: Mapping[str, Any]) -> bytes:
    """RFC 8785 JCS: UTF-8, no whitespace, keys sorted by UTF-16 code unit,
    ECMAScript Number::toString for numbers, \\u escapes only where required."""

EXCLUDED_FIELDS: frozenset[str] = frozenset({"evidence_hash", "ledger"})

def evidence_hash(doc: Mapping[str, Any]) -> str:
    """sha256(canonicalise(strip(doc, EXCLUDED_FIELDS))).hexdigest()"""

def assemble(alert, tracks, items, config_version, *, now) -> dict: ...
def verify(doc: Mapping[str, Any], expected_hash: str) -> VerificationResult: ...
```

- Hashing happens **in the worker, at assembly time** — not in the API, not in a trigger.
  The hash covers the object as the worker saw it.
- Every `evidence_item` file is itself SHA-256'd and its digest is *inside* the hashed
  document, so an altered JPEG invalidates the alert hash.
- `verify()` returns a structured diff on mismatch, never just `False`. Failing loudly is
  the entire point.
- Fixed vectors in `worker/tests/fixtures/jcs/` (from the RFC 8785 test suite) guard the
  canonicaliser, including the float edge cases (`1e21`, `-0`, `1e-7`) that are where every
  naive implementation breaks.

### 7.12 Ledger — `ledger/` **[DEMO-CRITICAL interface, STRETCH backend]**

Blocker #6. Build the mock first; wire everything to the interface; Fabric is a backend
swap, never a dependency of the critical path.

```python
class LedgerBackend(Protocol):
    def anchor(self, merkle_root: str, meta: Mapping[str, Any]) -> AnchorReceipt: ...
    def get(self, tx_id: str) -> AnchorRecord | None: ...
    def health(self) -> LedgerHealth: ...

class MockLedger(LedgerBackend):   # append-only JSONL + SHA-256 chain, on disk
class FabricLedger(LedgerBackend): # Fabric Gateway, channel 'ibvap', cc 'evidencecc'
```

Merkle (`merkle.py`, pure, property-tested):

```python
def build_tree(leaves: Sequence[str]) -> MerkleTree: ...
def proof(tree: MerkleTree, index: int) -> list[tuple[str, str]]:  # [(side, hash)]
def verify_proof(leaf: str, proof, root: str) -> bool: ...
```

- Domain separation: leaves hashed `0x00 || leaf`, internal nodes `0x01 || l || r`. Without
  it, second-preimage attacks on the tree are textbook.
- Odd node count duplicates the last node — and `hypothesis` tests every tree size 1..257
  because that is where off-by-one lives.
- **Anchoring never blocks alerting (Invariant).** `alert.ledger_status='pending'` is a
  perfectly good alert. `ibvap-anchor` drains the queue every `anchor_interval_s`
  (default 30) and retries with backoff forever. `make demo` runs with `backend=mock`.

### 7.13 Sinks — `sinks.py`

```python
class AlertSink(Protocol):
    def emit(self, alert: AlertRecord) -> None: ...

class PostgresSink:  # single INSERT ... RETURNING, one txn, alert + evidence_items
class RedisSink:     # XADD ibvap:alerts, capped at 10k
class MinioSink:     # put_object for snapshot/clip, returns keys + sha256
class NullSink:      # tests
```

All sinks are **fail-soft**: a sink raising is logged with full context and the pipeline
continues. Losing the Redis publish must not lose the DB row. Sink failures increment
`ibvap_sink_failures_total{sink=…}` and turn the dashboard's health dot amber.

### 7.14 Pipeline — `pipeline.py`

Owns thread topology and backpressure.

```
RtspReader ──▶ Queue[Frame] (maxsize=4, DROP-OLDEST) ──▶ InferenceThread (shared, batched)
                                                              │
                                                              ▼
                                              Queue[DetBundle] (maxsize=16, BLOCK)
                                                              │
                                                              ▼
                        StageThread(per camera): track → geometry → rules → risk
                                        → debounce → evidence → sinks
```

- **Frame queue drops oldest.** On a slow host we analyse *recent* frames, never a growing
  backlog of stale ones. Every drop increments a counter; sustained drops log at WARNING
  and show in the dashboard as a reduced effective fps. Being honest about degradation is
  better than lying with latency.
- **Detection queue blocks** — a full detection queue means the stage thread is wedged,
  which is a bug we want to see, not paper over.
- Evidence clip writing is handed to a small `ThreadPoolExecutor`, never inline.
- `Pipeline.health()` returns per-camera state for `/api/v1/health` and `stream_health`.
- Every thread has a name (`reader-BOP03-CAM01`) so stack dumps are readable at 2 a.m.

---

## 8. API surface — `api/`

FastAPI, async, all routes under `/api/v1`. JWT bearer (HS256, local key), refresh via
rotating token. RBAC enforced by a dependency, never by the router body.

| Method | Path | Role | Notes |
| --- | --- | --- | --- |
| `POST` | `/auth/login` | — | argon2id; rate-limited 5/min/IP |
| `POST` | `/auth/refresh` | any | rotates |
| `GET` | `/sites` | viewer | |
| `GET` | `/cameras` | viewer | `?site_id=` |
| `GET` | `/cameras/{id}/stream` | viewer | returns HLS URL + MediaMTX path |
| `GET/POST/PUT/DELETE` | `/cameras/{id}/zones` | admin (write) | normalised polygons |
| `GET` | `/alerts` | viewer | filters: `site_id, camera_id, severity, kind, status, from, to`; keyset pagination on `(ts_utc, id)` |
| `GET` | `/alerts/{id}` | viewer | full `evidence_doc` + `risk_breakdown` |
| `POST` | `/alerts/{id}/ack` | operator | |
| `POST` | `/alerts/{id}/adjudicate` | operator | `{verdict, note}` — feeds `make eval` |
| `GET` | `/alerts/{id}/evidence/{item_id}` | investigator | presigned MinIO URL, 5 min TTL, writes `audit_log` |
| `GET` | `/alerts/{id}/verify` | viewer | **[DEMO-CRITICAL]** recomputes the hash, checks the Merkle proof, queries the ledger, returns every intermediate value |
| `POST` | `/verify/document` | viewer | verify an *uploaded* evidence JSON against the ledger — the "prove it to a sceptic" endpoint |
| `GET` | `/watchlist/vehicles` · `POST` | investigator | plate submitted plaintext, stored HMAC'd, never echoed |
| `GET` | `/watchlist/persons` · `POST` | admin | 403 when faces disabled |
| `GET` | `/health` | — | worker/stream/db/minio/redis/ledger |
| `GET` | `/metrics` | — | Prometheus text |
| `WS` | `/ws/alerts` | viewer | §9 |

**`/alerts/{id}/verify` response** — the whole blockchain story in one payload:

```json
{
  "alert_id": "…", "stored_hash": "9f2c…", "recomputed_hash": "9f2c…",
  "hash_match": true,
  "canonical_bytes_sha256": "9f2c…", "canonical_length": 2841,
  "evidence_items": [{"kind":"snapshot","object_key":"…","sha256":"ab…","file_matches":true}],
  "merkle": {"leaf_index": 17, "leaf_hash": "…", "proof": [["L","…"],["R","…"]],
             "computed_root": "7a…", "stored_root": "7a…", "root_match": true},
  "ledger": {"backend":"fabric","tx_id":"…","block_number":412,
             "anchored_at":"…","root_on_chain":"7a…","chain_match":true},
  "verdict": "VERIFIED",
  "checks": [{"name":"evidence_hash","passed":true}, …]
}
```

`verdict ∈ {VERIFIED, PENDING_ANCHOR, TAMPERED, UNVERIFIABLE}`. `TAMPERED` includes a
field-level diff. Loud failure, P5.

---

## 9. Realtime contract — `WS /ws/alerts`

Server→client envelope, discriminated on `type`:

```ts
type ServerMsg =
  | { type: "hello";        server_time: string; subscribed: string[] }
  | { type: "alert";        alert: AlertSummary }
  | { type: "alert_update"; alert_id: string; status: string; adjudication?: string }
  | { type: "stream_health";camera_id: string; state: StreamState; fps: number }
  | { type: "evqm";         camera_id: string; profile: string; metrics: QualityMetrics }
  | { type: "ledger";       batch_id: string; status: string; tx_id?: string }
  | { type: "heartbeat";    ts: string };
```

- Client sends `{type:"subscribe", site_ids:[…], min_severity:"low"}` after connect.
- Heartbeat every 15 s; client reconnects with exponential backoff and **replays missed
  alerts via `GET /alerts?from=<last_seen_ts>`**. A dropped socket must never mean a lost
  alert on screen.
- Fan-out reads Redis Streams with a consumer group per API instance.

---

## 10. Dashboard — `dashboard/`

React 18 + Vite + TS strict + Tailwind. Four screens.

1. **Live Wall** — 1/4/9/16-up HLS via hls.js, zone overlays drawn on canvas, live boxes
   from the WS feed, per-camera health dot + EVQM profile chip.
2. **Alerts** — virtualised stream, severity colour, filters, keyboard-first
   (`j/k` move, `a` acknowledge, `Enter` open). Ack in one keystroke or an operator
   will not ack at all.
3. **Alert detail** — snapshot, clip, **the risk breakdown rendered as a waterfall** with
   negative contributions in a different colour (P2 made visible), reason codes, track
   path, and the verification panel.
4. **Verify** — drop an evidence JSON, see every check, green or red, with the computed
   values on screen. This is the panel that wins the blockchain argument.

Plus **Admin**: cameras, zone editor (draw polygons/tripwires on a paused frame),
watchlists, users, and a config-diff view.

Design constraints: dark theme (control-room lighting), colour never the sole severity
carrier (accessibility), every timestamp shows local + UTC on hover, and the whole app
must be usable at 1280×720 because that is what the projector will be.

---

## 11. Configuration — `config/`

```
config/
  defaults.yaml      # everything, with documented defaults
  detector.yaml      # backend, weights, input size, per-class thresholds, CLASS_MAP
  evqm.yaml          # metric thresholds, hysteresis, sampling
  rules.yaml         # per-rule enable + params
  risk.yaml          # weights, severity bands
  anpr.yaml          # regions, voting, HMAC key ref
  faces.yaml         # enabled: false
  ledger.yaml        # backend: mock, anchor interval
  profiles/{laptop,bop,edge}.yaml
  sites/{site_code}.yaml
```

Merge order (later wins): `defaults` → `profiles/<profile>` → `sites/<site>` → env
(`IBVAP__RULES__LOITER__SECONDS=45`) → CLI flags. Loaded once at start into a frozen
Pydantic model; `config_version` = SHA-256 of the merged document, written to every alert.
A running worker never reloads config silently — a change means a restart and a new
`config_version`, so every alert is attributable to an exact ruleset.

Secrets (`DB_PASSWORD`, `MINIO_SECRET_KEY`, `JWT_SECRET`, `PLATE_HMAC_KEY`,
`RTSP_CREDENTIALS`) come from env or a mounted file — never from `config/`, never from git.
`make demo` generates throwaway ones into `.env` on first run.

---

## 12. Security and privacy

- **At rest:** RTSP credentials and plate ciphertext encrypted with AES-256-GCM under a
  key from env. Plate lookup uses HMAC-SHA256 so the DB holds no reversible identifier.
- **In transit:** TLS on API and MinIO in the `bop` profile; self-signed is acceptable at a
  BOP and the dashboard pins the fingerprint.
- **RBAC:** `viewer` (see alerts, no media) < `operator` (+ ack/adjudicate, live video) <
  `investigator` (+ evidence download, plate reveal) < `admin` (+ config, watchlists, users).
- **Audit:** every evidence access, plate reveal, watchlist change, config change and login
  writes `audit_log`. The audit log is itself Merkle-anchored daily.
- **Data minimisation (P6):** faces off by default; non-matching embeddings destroyed at
  track close; snapshots cropped to the region of interest where policy requires it;
  retention enforced by a nightly job, not by hope.
- **Threat model we actually defend:** an insider editing the alerts table to make an
  incident disappear. The Merkle root is anchored externally, so a deleted or altered alert
  is detectable — the row's hash no longer reproduces, or its leaf is missing from a batch
  whose root is already on chain.
- **Threat we do not defend:** an attacker with root on the worker host at capture time.
  Nothing downstream can fix a lie told at the source; we say so rather than implying
  otherwise.

---

## 13. Failure modes and degradation

| Failure | Detection | Behaviour |
| --- | --- | --- |
| RTSP drop | Watchdog, 5 s | `STALLED` → reconnect w/ backoff; MediaMTX keeps recording (P8); dashboard dot red |
| Camera tamper (covered/moved) | EVQM step change | `CAMERA_TAMPER` alert — the surveillance system watching itself |
| GPU OOM | ORT/TRT exception | Fall back to CPU + smaller input size, log at ERROR, keep running |
| Postgres down | Sink exception | Alerts buffered to a local SQLite WAL spool, replayed on recovery. **Never drop an alert.** |
| MinIO down | Sink exception | Evidence spooled to local disk, uploaded on recovery; alert still fires with `evidence_pending` |
| Redis down | Sink exception | WS fan-out degrades to 5 s polling; DB path unaffected |
| Fabric down | `health()` | `ledger_status='pending'`; anchor queue drains later (Invariant) |
| Worker crash | Systemd/Compose restart | MediaMTX recording unaffected (P8); on restart, cameras resume, tracks do not |
| Clock skew | NTP check at start | Refuse to start if > 2 s off — evidence timestamps must be trustworthy |
| Disk full | Pre-flight + 60 s check | Stop analytics, keep recording, alert the operator |

---

## 14. Performance targets

| Metric | `laptop` | `bop` |
| --- | --- | --- |
| Cameras | 2 | 8–12 |
| Analytics fps/camera | 6 | 12 |
| Detection latency p95 | 120 ms | 45 ms |
| Glass-to-banner p95 | 900 ms | 500 ms |
| Alert precision (eval set) | ≥ 0.85 | ≥ 0.90 |
| Person recall @ ≥40 px height | ≥ 0.80 | ≥ 0.88 |
| Plate accuracy (exact, ≥80 px wide) | ≥ 0.75 | ≥ 0.85 |
| False alerts / camera / hour (idle scene) | ≤ 0.5 | ≤ 0.2 |
| Cold start to first alert | ≤ 45 s | ≤ 30 s |
| RSS, worker | ≤ 2.5 GB | ≤ 6 GB |

`make bench` measures the first five and writes `docs/BENCH.md`. Numbers in a pitch deck
that were not produced by `make bench` do not go in the pitch deck.

---

## 15. Evaluation — `make eval`

- **Eval set:** ≥ 40 clips in `eval/clips/` with `eval/labels.jsonl`
  (`{clip, t_start, t_end, expect: [{kind, camera, zone}]}`), covering day, dusk, night,
  fog, rain-on-lens, crowd, vehicle, empty-scene (the negatives matter most), and two
  deliberate tamper clips.
- **Metrics:** per-rule precision/recall, alert latency from ground-truth onset, false
  alerts per idle hour, plate exact-match rate, tracking ID-switch count.
- Operator adjudications from `/alerts/{id}/adjudicate` feed back as labels — the demo
  gets better the more it is used, which is also a nice thing to show a judge.

---

## 16. Build phases

Each phase lands with tests and a demo-able increment. **Do not start a phase before the
previous one's acceptance criteria pass.**

| # | Phase | Deliverable | Acceptance |
| --- | --- | --- | --- |
| **0** | Foundation | Repo, `make up`, compose (pg/minio/redis/mediamtx), config loader, logging, CI, `make models` | `make up && make test` green on a clean machine; `models/MANIFEST.json` verifies |
| **1** | Data & API skeleton | Migrations for §6, SQLAlchemy models, auth + RBAC, `/health`, `/sites`, `/cameras`, `make seed` | `make migrate seed` then `GET /cameras` returns 2 seeded cameras; RBAC denies a viewer writing a zone |
| **2** | Ingest **[DC]** | §7.1 reader + watchdog + reconnect, MediaMTX fixture streams, HLS out, `stream_health` | Kill the RTSP source mid-run: state goes LIVE→STALLED→RECONNECTING→LIVE, no thread leak, no hang. HLS plays in a browser |
| **3** | Detect + track **[DC]** | §7.4 + §7.5, warmup, batching, `FrameTransform` mapping, mock backend | Fixture clip yields stable `track_id`s; boxes land in original coords; warmup logged; ≥6 fps on CPU laptop profile |
| **4** | EVQM + enhance | §7.2 + §7.3, hysteresis, profile chip in health | Headlight-sweep fixture does **not** flip profile; day→night fixture switches within 3 samples |
| **5** | Rules + risk + alerts **[DC]** | §7.6–7.8, zones, debounce, alert rows, WS fan-out | Person-on-tripwire fixture yields **exactly one** alert per crossing; risk breakdown sums to score; 10-minute idle fixture yields 0 alerts |
| **6** | Evidence **[DC]** | §7.11, snapshot + clip to MinIO, JCS hashing, `/verify` | RFC 8785 vectors pass; a byte edit to the snapshot flips `verify` to TAMPERED with a diff |
| **7** | Ledger | §7.12 mock backend, Merkle, anchor service, verify panel | Proofs verify for tree sizes 1..257; killing the ledger leaves alerts firing with `pending` |
| **8** | ANPR | §7.9, plate detect + OCR + voting + HMAC | Fixture vehicle clip reads the plate correctly ≥3 frames; DB holds no plaintext |
| **9** | Dashboard **[DC]** | §10 live wall, alerts, detail w/ waterfall, verify panel | End-to-end: intrusion on fixture → banner < 1 s → open detail → verify green |
| **10** | Faces (opt-in) | §7.10, pgvector index, admin enrolment | Disabled by default; `test_faces_no_silent_enrolment` passes; enabling requires an audited admin action |
| **11** | Fabric | `make fabric-up`, chaincode, `FabricLedger` | Same proofs verify against the chain; falling back to mock is a one-line config change |
| **12** | Harden | `make bench`, `make eval`, `make demo` cold start, runbook | `make demo` works offline from a cold boot in < 3 min, twice in a row |

[STRETCH], in priority order: multi-camera hand-off, thermal/IR fusion, drone/UAV class,
audio gunshot detection, cross-BOP re-ID, mobile operator app, federated model updates.

---

## 17. Glossary

**BOP** Border Out Post · **EVQM** Environmental & Video Quality Monitor (§7.2) ·
**JCS** JSON Canonicalisation Scheme, RFC 8785 · **Tripwire** a line, crossing it in a
direction is an event · **Zone** a polygon on the image plane · **Foot-point** bottom
centre of a box, the ground-contact proxy · **Debounce** suppression of repeat alerts for
the same (camera, track, zone, rule) · **Anchor** writing a Merkle root to the ledger ·
**Profile** EVQM-selected processing mode, or a deployment size (§5) — disambiguated by
context.
