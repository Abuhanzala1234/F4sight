# Running F4SIGHT

One command, from a terminal:

```bash
wsl.exe -d Ubuntu -e bash -lc "cd '/mnt/c/Users/abu hanzala/F4sight-2' && PROFILE=laptop-gpu make demo"
```

Then open **http://localhost:8000/** and sign in with the demo credentials
(printed at the end of the seed step, also re-printable with `make seed`).

Drop `PROFILE=laptop-gpu` to run on CPU only (`make demo` alone) — the GPU
profile needs `onnxruntime-gpu[cuda,cudnn]` installed once
(`.venv/bin/pip install 'onnxruntime-gpu[cuda,cudnn]'`) and a CUDA-capable
GPU; it falls back to CPU on its own if either isn't there.

## Prerequisite

**Docker Desktop must already be running.** If it isn't, `make demo` fails
at its `up` step trying to reach the Docker daemon — that's the one step
that actually errors out rather than just being slow.

## What to expect

If Docker Desktop is already running and its containers from a previous
session are still up, this is genuinely one command — no follow-up needed.

If the machine has rebooted (or the containers got recreated) since you
last ran it:

- The app itself still comes up fine.
- MediaMTX's dynamic camera paths don't survive that, even though the
  database still marks a previously-connected camera as "enabled" — its
  tile will sit on **no signal** until you reconnect it.
- **Fix:** on the Live Wall, disconnect that camera (✕ on hover) and
  reconnect it with the same phone IP. That's the only "extra" step, and
  it's not an error — MediaMTX just doesn't persist dynamic state across a
  full container restart.

## Connecting a real camera

Any empty ("Connect camera") tile: click it, type the phone's IP (same
WiFi, IP Webcam app open, screen unlocked). Live within a few seconds — no
restart, the detector is already loaded and shared across every camera.

A phone's RTSP server generally serves **one client well** — splitting one
phone across two camera slots at once will cause visible lag/packet loss on
both. Use one phone per slot where possible.

## Stopping it

`Ctrl+C` in the same terminal — stops the worker, API and anchor together.
`docker compose down` afterward if you also want to stop
Postgres/MinIO/Redis/MediaMTX.

## More detail

- Full quick-start and architecture: [`../README.md`](../README.md)
- Incident playbooks: [`RUNBOOK.md`](RUNBOOK.md)
- The pitch deck and talk-track script: [`pitch/`](pitch/)
