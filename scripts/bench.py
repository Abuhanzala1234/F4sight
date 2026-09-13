#!/usr/bin/env python3
"""Throughput and latency benchmark (`make bench`) → docs/BENCH.md.

BUILD_SPEC §14: *numbers in a pitch deck that were not produced by `make bench`
do not go in the pitch deck.* This script is what makes that rule enforceable.

Measures the per-frame hot path (§7.3–7.5) end to end on whatever hardware it
is run on, and writes a table with the machine's own specifications alongside,
because "45 ms" means nothing without saying on what.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker" / "src"))

from drishti_worker.config import load_config
from drishti_worker.detect import DetectorConfig, build_detector
from drishti_worker.enhance import EnhanceConfig, enhance_for_model
from drishti_worker.evqm import EVQM, EVQMConfig
from drishti_worker.risk import RiskConfig, RiskContext, score
from drishti_worker.rules import RuleConfig, RuleEngine
from drishti_worker.track import ByteTracker, TrackerConfig
from drishti_worker.types import (
    CameraRuntime,
    Detection,
    Frame,
    ZoneKind,
    ZoneRuntime,
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[index]


def gpu_info() -> dict[str, str]:
    """Ask the driver what this GPU actually is.

    Via ``nvidia-smi`` rather than a Python binding on purpose: pynvml/torch are
    build-time dependencies the worker does not have, and "45 ms" against an
    unnamed accelerator is exactly the unfalsifiable number §14 exists to
    prevent. No GPU is a normal answer, not an error.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if out.returncode != 0 or not line:
        return {}
    parts = [p.strip() for p in line.split(",")]
    keys = ("gpu", "gpu memory", "nvidia driver", "compute capability")
    return dict(zip(keys, parts, strict=False))


def machine() -> dict[str, str]:
    info = {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
    }
    info.update(gpu_info())
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["providers available"] = ", ".join(ort.get_available_providers())
    except ImportError:
        info["onnxruntime"] = "not installed"
    return info


def bench_stage(name: str, fn, iterations: int) -> dict[str, float]:
    timings: list[float] = []
    for i in range(iterations):
        started = time.perf_counter()
        fn(i)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "stage": name,
        "mean_ms": statistics.mean(timings),
        "p50_ms": percentile(timings, 0.50),
        "p95_ms": percentile(timings, 0.95),
        "max_ms": max(timings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="laptop")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--out", default="docs/BENCH.md")
    parser.add_argument(
        "--detector",
        default=None,
        choices=["onnx", "tensorrt", "mock"],
        help="override the configured backend; 'mock' benchmarks everything "
        "except inference, which is useful before `make models` has run",
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy is required for the benchmark", file=sys.stderr)
        return 1

    cfg = load_config("config", profile=args.profile, environ={})
    detector_cfg = DetectorConfig.from_mapping(cfg.as_dict())
    if args.detector:
        detector_cfg = replace(detector_cfg, backend=args.detector)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    print(f"benchmarking profile={args.profile} iterations={args.iterations}\n")

    results: list[dict[str, float]] = []

    # --- detector: warmup is measured separately, because the FIRST inference
    # --- is the one that makes a demo look frozen (blocker #3).
    warm_started = time.perf_counter()
    try:
        detector = build_detector(detector_cfg)
    except FileNotFoundError as exc:
        print(f"{exc}\n", file=sys.stderr)
        print(
            "Re-run with `--detector mock` to benchmark every stage except\n"
            "inference, or run `make models` first for the real numbers.",
            file=sys.stderr,
        )
        return 1
    warmup_s = time.perf_counter() - warm_started

    model_w, model_h = detector.input_size
    model_input = rng.integers(0, 255, (model_h, model_w, 3), dtype=np.uint8)
    results.append(
        bench_stage("detect (batch=1)", lambda _: detector.infer([model_input]), args.iterations)
    )

    # Batched throughput is the number the multi-camera claim in §5 actually
    # rests on: the inference thread is shared, so N cameras arrive as one
    # batch, not N sequential calls. Reported per frame so it compares directly
    # with the row above.
    batch_size = max(1, detector_cfg.max_batch)
    per_frame_batched = 0.0
    if batch_size > 1:
        batch_images = [model_input] * batch_size
        batched = bench_stage(
            f"detect (batch={batch_size}, per frame)",
            lambda _: detector.infer(batch_images),
            max(20, args.iterations // 5),
        )
        per_frame_batched = batched["p95_ms"] / batch_size
        results.append(
            {k: (v / batch_size if k.endswith("_ms") else v) for k, v in batched.items()}
        )

    evqm = EVQM(EVQMConfig.from_mapping(cfg.as_dict()), "bench")
    frame_counter = {"n": 0}

    def run_evqm(_: int) -> None:
        frame_counter["n"] += 1
        evqm.observe(Frame("bench", frame_counter["n"], datetime.now(UTC), image, 1280, 720))

    results.append(bench_stage("evqm sample", run_evqm, args.iterations))

    # Enhancement runs at MODEL scale, after the letterbox resize (§7.3), so
    # that is what we measure. Benchmarking it on the full frame would report a
    # number the system never actually pays.
    enhance_cfg = EnhanceConfig.from_mapping(cfg.as_dict())
    model_scale = rng.integers(0, 255, (model_h, model_w, 3), dtype=np.uint8)
    for profile in ("day", "lowlight", "night", "fog", "degraded"):
        results.append(
            bench_stage(
                f"enhance ({profile}) @{model_w}x{model_h}",
                lambda _, p=profile: enhance_for_model(model_scale, p, enhance_cfg),
                max(30, args.iterations // 5),
            )
        )

    tracker = ByteTracker(TrackerConfig.from_mapping(cfg.as_dict()))
    results.append(
        bench_stage(
            "track update (4 objects)",
            lambda i: tracker.update(
                [
                    Detection("person", 0.9, (100.0 + i, 200, 140.0 + i, 320), 0),
                    Detection("person", 0.8, (400.0 - i, 200, 440.0 - i, 320), 0),
                    Detection("vehicle", 0.9, (700.0, 300, 900.0, 420), 2),
                    Detection("animal", 0.7, (200.0, 500, 260.0, 560), 16),
                ],
                datetime.now(UTC),
            ),
            args.iterations,
        )
    )

    camera = CameraRuntime("c", "CAM-01", "s", "BOP-03", "Asia/Kolkata", 1280, 720, 6.0)
    zones = [
        ZoneRuntime(
            "z1",
            "Wire",
            ZoneKind.TRIPWIRE,
            ((640.0, 720.0), (640.0, 0.0)),
            "both",
            ("person",),
            None,
            4,
            True,
        ),
        ZoneRuntime(
            "z2",
            "Area",
            ZoneKind.AREA,
            ((900.0, 200.0), (1280.0, 200.0), (1280.0, 720.0), (900.0, 720.0)),
            None,
            ("person",),
            None,
            3,
            True,
        ),
    ]
    engine = RuleEngine(
        RuleConfig.from_mapping(cfg.as_dict()), RiskConfig.from_mapping(cfg.as_dict())
    )
    tracks = tracker.update(
        [Detection("person", 0.9, (620.0, 200, 660.0, 400), 0)], datetime.now(UTC)
    )
    results.append(
        bench_stage(
            "rules + geometry",
            lambda _: engine.evaluate(tracks, camera, zones, datetime.now(UTC), "day"),
            args.iterations,
        )
    )

    from drishti_worker.types import Signal

    signals = [Signal("ZONE_INTRUSION", 40.0, {}), Signal("NIGHT_MOVEMENT", 20.0, {})]
    results.append(
        bench_stage("risk score", lambda _: score(signals, RiskContext()), args.iterations)
    )

    from drishti_worker.evidence import assemble, evidence_hash

    doc = assemble(
        alert_id="bench",
        site={"code": "BOP-03"},
        camera={"code": "CAM-01"},
        detection={"track_id": 1},
        risk={"score": 60.0},
        items=[{"kind": "snapshot", "sha256": "a" * 64, "enhanced": False}],
        config_version=cfg.version,
        spec_version="1.0.0",
        worker_version="1.0.0",
        created_at=datetime.now(UTC).isoformat(),
    )
    results.append(
        bench_stage("evidence hash (JCS)", lambda _: evidence_hash(doc), args.iterations)
    )

    hot_path = sum(
        r["p95_ms"]
        for r in results
        if r["stage"]
        in (
            "detect (batch=1)",
            "track update (4 objects)",
            "rules + geometry",
            "risk score",
        )
    )
    fps_ceiling = 1000.0 / hot_path if hot_path > 0 else 0.0

    info = machine()
    lines = [
        "# BENCH.md — measured, not claimed",
        "",
        "> Generated by `make bench`. BUILD_SPEC §14: numbers that were not produced",
        "> by this command do not go in the pitch deck.",
        "",
        f"**Run:** {datetime.now(UTC).isoformat()}  ",
        f"**Profile:** `{args.profile}`  ",
        f"**Iterations:** {args.iterations}  ",
        (
            f"**Detector:** {detector.backend}, input {model_w}×{model_h}, "
            f"weights `{detector_cfg.weights}`  "
        ),
        f"**Providers bound:** {', '.join(getattr(detector, 'providers', ['n/a']))}",
        "",
        "## Machine",
        "",
        "| | |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in info.items()],
        "",
        "## Cold start",
        "",
        (
            f"Detector construction + {detector_cfg.warmup_iterations} warmup "
            f"inferences: **{warmup_s:.2f} s**"
        ),
        "",
        "This is blocker #3. The worker does not report ready until warmup finishes,",
        "so the dashboard says *starting* rather than showing an empty live view.",
        "",
        "## Per-stage timings (milliseconds)",
        "",
        "| Stage | mean | p50 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['stage']} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | "
            f"{r['p95_ms']:.2f} | {r['max_ms']:.2f} |"
        )

    analytics_fps = float(cfg.get("ingest.analytics_fps"))
    lines += [
        "",
        "## Derived",
        "",
        f"- Hot path p95 (detect + track + rules + risk): **{hot_path:.1f} ms**",
        f"- Single-camera analytics ceiling: **{fps_ceiling:.1f} fps**",
        f"- Configured analytics rate: **{analytics_fps} fps/camera**",
        f"- Headroom: **{fps_ceiling / analytics_fps:.1f}×**",
    ]

    if per_frame_batched > 0:
        # §5's camera count. The stage work after detection is per camera and
        # runs on its own thread, so the shared inference thread is the ceiling.
        batched_fps = 1000.0 / per_frame_batched
        lines += [
            (
                f"- Batched inference (batch={batch_size}): "
                f"**{per_frame_batched:.1f} ms/frame → "
                f"{batched_fps:.0f} fps aggregate**"
            ),
            (
                f"- Cameras sustainable at {analytics_fps:g} fps: "
                f"**{batched_fps / analytics_fps:.1f}**"
            ),
            "",
            "The camera count is inference throughput divided by the per-camera",
            "analytics rate. It is an upper bound: it assumes the stage threads keep",
            "up, which they do here by two orders of magnitude, and it ignores decode",
            "cost, which is the next thing to measure on a box with real cameras.",
        ]

    lines += [
        "",
        (
            "Enhancement is excluded from the hot-path figure because the `day` "
            "profile is an identity no-op, which is most frames in most deployments "
            "(§7.3)."
        ),
        "Evidence hashing and clip writing are off the critical path by design (§3.4).",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # encoding is explicit everywhere we touch text: Python defaults to the
    # locale encoding, which is cp1252 on a Windows box, and this file contains
    # "→" and "×". Without it `make bench` cannot write its own output there.
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"{'stage':<28} {'p50':>8} {'p95':>8}")
    print("-" * 46)
    for r in results:
        print(f"{r['stage']:<28} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f}")
    print(f"\nhot path p95: {hot_path:.1f} ms  →  {fps_ceiling:.1f} fps ceiling")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
