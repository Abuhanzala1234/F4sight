# RUNBOOK.md — what to do when something is wrong

This is the "it's 2 a.m. and an alert isn't showing up" document. `docs/BUILD_SPEC.md`
§13 has the design-level failure table; this is the same information as a set of
things to actually *type*.

---

## Cold start

```bash
make demo
```

Cold, offline, from nothing running. If this doesn't get you to a working system in
under 3 minutes, work through in order:

1. `docker compose ps` — postgres, minio, redis, mediamtx all `healthy`? If not,
   `make down && make up` and check `make logs`.
2. `python scripts/fetch_models.py --verify-only` — model weights present and
   hash-correct? If not, `make models` (needs network, ~166 MB, once).
3. `make migrate` — did it apply cleanly? A stuck migration usually means the
   Postgres container isn't actually up yet; `make up` again.
4. Check the worker's own stderr line: `drishti-worker READY` has to appear.
   If the process exits before that, read the last log line — every startup
   failure in `__main__.py` (`check_clock`, `verify_models`, `build_detector`)
   raises `SystemExit` with the fix in the message, not just a traceback.

---

## Health checks, fastest to slowest

```bash
curl -s localhost:8000/api/v1/health | jq .        # every component, one call
curl -s localhost:8000/api/v1/health/live          # liveness only, for a load balancer
```

`health.components[]` — `name`, `ok`, `detail`. Read `detail` before doing anything
else; it is written to say what's actually wrong, not just that something is.

Per-camera state lives in the worker's own log line every `pipeline.stats_interval_s`
(default 10 s): `state=... profile=... fps_in=... tracks=... alerts=... dropped=...`.
`dropped > 0` and climbing means the frame queue can't keep up — see "Worker is slow"
below before assuming a camera problem.

---

## Incident playbooks

### A camera's dot is red / `STALLED` or `RECONNECTING`

This is the watchdog working, not broken. §7.1: `LIVE → STALLED (5s) → RECONNECTING
(backoff) → LIVE`. MediaMTX keeps recording throughout (P8) — you have not lost
footage. If it never comes back:

- Check the RTSP URL still resolves: `ffprobe rtsp://...` from the worker's own
  host, not your laptop. A camera behind NAT or a rebooted NVR is the common cause.
- Check `stream_health` in the DB / the health endpoint for the last error string.
- Restarting only the worker (`make worker`) does **not** need a MediaMTX restart —
  they are independent processes on purpose (§3.1).

### Alerts fire but the dashboard shows nothing

1. Is the WebSocket connected? `Alerts` page — the feed-live/feed-down dot in the
   header is not decorative. If down: check `/ws/alerts`, check Redis
   (`redis-cli ping`). Redis down degrades to 5 s polling per §13, not silence —
   if you see neither, the API process itself may be down.
2. `GET /alerts?from=<iso timestamp>` — is the alert actually in Postgres? If yes,
   it's a rendering/WS problem, not a pipeline one. If no, keep going down this list.
3. Check the worker log for `ALERT camera=... kind=...` — did it fire at all?
   If not, the rule's `gate()` may be rejecting the track (`min_hits`,
   `min_track_age_frames`, `min_track_conf` — logged at DEBUG per rejection).
4. Check `PostgresSink`/`RedisSink` for exceptions in the worker log. Both are
   fail-soft (§7.13): a failing sink logs and spools, it does not silently drop.
   `spool/alerts-*.jsonl` existing and growing means Postgres (or whichever sink)
   is down and alerts are queued, not lost. Replay with:
   ```bash
   python scripts/replay_spool.py --spool spool          # add --delete once confirmed
   ```

### `/alerts/{id}/verify` returns `TAMPERED`

Read the `checks[]` array — it names exactly which check failed, with computed vs.
stored values. Three real causes, in order of likelihood:

1. **A byte in MinIO actually changed.** `evidence_items[].file_matches: false`
   means the object's live SHA-256 no longer matches what's inside the hashed
   document. This is the system working as designed — say so, don't "fix" it by
   re-uploading.
2. **The evidence document itself was edited in the DB.** `hash_match: false` but
   every media file matches. Investigate who has write access to `alert.evidence_doc`
   — nothing in the normal request path does this.
3. A canonicalisation bug (should not happen; `evidence.py` has RFC 8785 vectors
   and 27 tests). If you hit this, it's a code bug, not an operational one —
   file it, don't paper over it.

`PENDING_ANCHOR` is not an incident. It means the Merkle root hasn't been anchored
to the ledger yet (mock or Fabric) — the alert is still fully verifiable against
its own stored hash. `ledger_status` on the alert row shows the anchor queue depth.

### Ledger (`drishti-anchor`) is down

Per the invariant table: **ledger unavailability never blocks alerting.** Alerts
keep firing with `ledger_status='pending'`. Bring the anchor service back
(`make anchor`, or `make fabric-up` if it's the Fabric backend that's down) and the
queue drains on its own — nothing needs to be replayed manually, unlike the sink
spool above.

### `make bench` reports 0 ms / absurd fps

You're benchmarking the mock detector. `docs/BENCH.md`'s header line says which
backend produced it — read that before trusting any number in the table (§14's own
rule: a number `make bench` didn't produce doesn't go in a pitch deck, and a number
it produced against `mock` doesn't either). Run `make models` first, or pass
`--detector onnx` explicitly once weights exist.

### GPU present but everything runs on CPU

Check the worker's own startup line: `onnx detector loaded weights=... providers=...`.
If it says `CPUExecutionProvider` on a box with an NVIDIA GPU:

1. `nvidia-smi` — does the driver see the card at all?
2. Check the log a few lines up for a provider creation error
   (`onnxruntime.preload_dlls()` runs before session creation specifically to avoid
   this — see `detect/onnx_yolo.py`'s `_preload_gpu_libraries`). The common one on
   Windows: `cublasLt64_13.dll` missing means the `nvidia-*` CUDA wheels aren't
   installed (`pip install "onnxruntime-gpu[cuda,cudnn]"`), not that the driver is
   broken.
3. TensorRT specifically needs `nvinfer_10.dll` — ONNX Runtime 1.30 links TensorRT
   10, and at the time this was written TensorRT 10 publishes no Python 3.14
   wheels (only 11, which ORT doesn't yet link). CUDA alone is not blocked by this;
   TensorRT is. See README Gate 5.
4. If none of that explains it: the provider chain falls back to CPU on *any*
   session-creation failure and logs the full exception first (P8) — read that
   exception, it names the actual missing library.

### A watchlisted plate isn't raising an alert

1. Is `anpr.enabled` true and is `PLATE_HMAC_KEY` set (>= 16 bytes) in the
   worker's environment? Both are logged at startup (`build_anpr` in
   `__main__.py`) if either is missing — check for the warning, not silence.
2. Is the vehicle track actually gated? Same `min_track_age_frames` gate as
   everything else (§7.7.1) — a plate can settle (3 frames of agreement, by
   default) well before the track is old enough for *any* rule to fire, watchlist
   included. This is expected, not a bug: the hit is cached and re-offered every
   frame after it settles specifically so it's still there once the gate opens.
3. Is the watchlist cache actually populated? `WatchlistCache.plate_count` — if
   Postgres was down when the worker started, the first successful refresh
   (default every 60 s, `anpr.watchlist_refresh_s`) is what populates it; there is
   no manual nudge needed once the DB comes back.
4. Remember: the DB never holds the plate text, only its HMAC (P6). If you're
   checking "is this plate on the list" by eye against `watchlist_vehicle`, you
   can't — compute the HMAC with the same key and compare, or use the API.

### Worker is slow / frame queue dropping

`frame_queue has dropped N frames since start` in the log means the shared
inference thread can't keep up with all cameras combined. In order of effort:

1. Confirm you're not accidentally on the `mock` or CPU path (see above) —
   this is by far the most common cause and costs zero config changes to fix.
2. Lower `ingest.analytics_fps` — the system is explicitly designed to
   degrade this way (§3.4's "recent frames beat a growing backlog").
3. Check `detector.batching.max_batch` / `max_wait_ms` — a `bop`-profile
   value on `laptop` hardware holds frames longer than the box can clear them.
4. `make bench` on the actual box, with real weights, and compare the
   "cameras sustainable" line in `docs/BENCH.md` against how many you're
   actually running.

---

## Rollback

There is no automated rollback (P7: no automated response action — the same
principle applies to the system itself). To roll back a bad worker deploy:

```bash
git log --oneline -- worker/          # find the last good commit
git checkout <sha> -- worker/
make test                             # confirm before restarting
# restart the worker process (systemd/compose, whichever runs it)
```

The database schema is append-only forward via Alembic; there is no
`alembic downgrade` path exercised in this project — a schema rollback needs a
human reading the specific migration, not a script.

---

## Things that are not incidents

- `ledger_status='pending'` on a fresh alert. Ask again in a few seconds.
- A camera in `CONNECTING` for the first few seconds after `make demo`'s cold
  start — the detector warmup (blocker #3) runs before any camera is even asked
  to connect, and MediaMTX/RTSP handshake takes a moment on top of that.
- `TensorrtExecutionProvider` absent from `providers available` in `make bench`'s
  machine table on a non-NVIDIA box, or on Python 3.14 anywhere right now (see
  above). CUDA or CPU picking up the work instead is the fallback chain doing its
  job, not a fault.
