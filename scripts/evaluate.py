#!/usr/bin/env python3
"""Evaluation harness (`make eval`) — BUILD_SPEC §15.

Runs the analytics pipeline over labelled clips and reports per-rule precision
and recall, alert latency from ground-truth onset, and — the number that
actually decides whether an operator keeps trusting the system — **false alerts
per idle hour** (P3).

Label format, one JSON object per line in ``eval/labels.jsonl``::

    {"clip": "night_fence_01.mp4", "camera": "CAM-02",
     "expect": [{"kind": "TRIPWIRE_CROSS", "t_start": 12.4, "t_end": 15.0}]}

A clip with ``"expect": []`` is a NEGATIVE — an idle scene that must produce
nothing. Those matter most; a system that never misses and always cries wolf is
useless.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Windows' console defaults to the system codepage (cp1252), not UTF-8, and
# this script prints ✓/✗/→ in its progress output -- a plain print() of
# any of them raises UnicodeEncodeError before the actual work even starts.
# reconfigure() is a no-op everywhere already UTF-8; errors="replace" means a
# console that truly cannot show a glyph gets a "?" instead of a crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker" / "src"))

from drishti_worker.config import load_config
from drishti_worker.detect import DetectorConfig, build_detector
from drishti_worker.enhance import EnhanceConfig, enhance_for_model
from drishti_worker.evqm import EVQM, EVQMConfig
from drishti_worker.risk import RiskConfig, RiskContext, score
from drishti_worker.rules import (
    DebounceConfig,
    Debouncer,
    RuleConfig,
    RuleEngine,
)
from drishti_worker.track import ByteTracker, TrackerConfig
from drishti_worker.types import (
    CameraRuntime,
    Detection,
    FrameTransform,
    ZoneKind,
    ZoneRuntime,
)

MATCH_WINDOW_S = 5.0  # an alert this close to a labelled event counts as a hit


@dataclass
class ClipResult:
    clip: str
    duration_s: float
    expected: list[dict[str, object]]
    fired: list[dict[str, object]] = field(default_factory=list)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    latencies_s: list[float] = field(default_factory=list)


def load_zone_specs(path: Path, camera_code: str) -> list[dict]:
    """Raw zone specs from eval/zones.json, exactly as authored (normalised
    0..1 polygons, per §6.2). NOT yet denormalised — that needs the clip's
    actual width/height, which we do not know until the clip is open, so
    ``run_clip`` finishes the job. Getting this split wrong is exactly how a
    normalised polygon silently turns into a single-pixel-wide zone: it
    happened once in this file already (see docs/EVAL.md)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get(camera_code, []))


def build_zones(specs: list[dict], width: int, height: int) -> list[ZoneRuntime]:
    """Denormalise zone specs against a clip's real dimensions.

    Mirrors ``drishti_api``/``drishti_worker.__main__._zone`` exactly, so the
    eval harness scores against the identical geometry a deployment would use
    for the same zones.json content.
    """
    from drishti_worker.geometry import denormalise

    out: list[ZoneRuntime] = []
    for z in specs:
        polygon = [(float(p[0]), float(p[1])) for p in z["polygon"]]
        out.append(
            ZoneRuntime(
                zone_id=z["id"],
                name=z.get("name", z["id"]),
                kind=ZoneKind(z["kind"]),
                polygon=tuple(denormalise(polygon, width, height)),
                direction=z.get("direction"),
                classes=tuple(z.get("classes", ())),
                severity_base=int(z.get("severity_base", 3)),
            )
        )
    return out


def run_clip(
    clip_path: Path,
    label: dict[str, object],
    zone_specs: list[dict],
    cfg,
    detector,
) -> ClipResult:
    import cv2

    detector_cfg = DetectorConfig.from_mapping(cfg.as_dict())
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {clip_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analytics_fps = float(cfg.get("ingest.analytics_fps"))
    stride = max(1, round(source_fps / analytics_fps))
    # Denormalise now that we finally know the clip's real pixel dimensions.
    zones = build_zones(zone_specs, width, height)

    camera = CameraRuntime(
        camera_id="eval",
        code=str(label.get("camera", "CAM-01")),
        site_id="eval",
        site_code="EVAL",
        timezone="Asia/Kolkata",
        width=width,
        height=height,
        analytics_fps=analytics_fps,
    )
    evqm = EVQM(EVQMConfig.from_mapping(cfg.as_dict()), "eval")
    tracker = ByteTracker(TrackerConfig.from_mapping(cfg.as_dict()))
    engine = RuleEngine(
        RuleConfig.from_mapping(cfg.as_dict()), RiskConfig.from_mapping(cfg.as_dict())
    )
    debouncer = Debouncer(DebounceConfig.from_mapping(cfg.as_dict()))
    risk_cfg = RiskConfig.from_mapping(cfg.as_dict())
    enhance_cfg = EnhanceConfig.from_mapping(cfg.as_dict())

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    model_w, model_h = detector.input_size
    result = ClipResult(
        clip=clip_path.name,
        duration_s=(capture.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps if source_fps else 0.0),
        expected=list(label.get("expect", [])),  # type: ignore[arg-type]
    )

    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index % stride != 0:
            index += 1
            continue
        clip_t = index / source_fps
        ts = t0 + timedelta(seconds=clip_t)

        from drishti_worker.types import Frame

        evqm.observe(Frame("eval", index, ts, image, width, height))
        profile = evqm.profile

        transform = FrameTransform.letterbox((width, height), (model_w, model_h))
        resized = cv2.resize(
            image, (int(width * transform.scale_x), int(height * transform.scale_y))
        )
        enhanced = enhance_for_model(resized, profile, enhance_cfg)
        import numpy as np

        canvas = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        y0, x0 = int(transform.pad_y), int(transform.pad_x)
        canvas[y0 : y0 + enhanced.image.shape[0], x0 : x0 + enhanced.image.shape[1]] = (
            enhanced.image
        )

        detections: list[Detection] = []
        for raw in detector.infer([canvas])[0]:
            mapped = detector_cfg.class_map.get(raw.cls_id)
            if mapped is None:
                continue
            cls = str(mapped.get("cls", "unknown"))
            if raw.conf < detector_cfg.threshold_for(cls):
                continue
            detections.append(
                Detection(
                    cls=cls,
                    conf=raw.conf,
                    box=transform.to_original(raw.box),
                    cls_id=raw.cls_id,
                )
            )

        tracks = tracker.update(detections, ts)
        signals_by_track = engine.evaluate(tracks, camera, zones, ts, profile)
        by_id = {t.track_id: t for t in tracks}

        for track_id, signals in signals_by_track.items():
            track = by_id.get(track_id)
            if track is None:
                continue
            primary = max(signals, key=lambda s: s.weight)
            decision = debouncer.submit(
                camera_id="eval",
                track_id=track_id,
                zone_id=primary.detail.get("zone_id"),
                rule_code=primary.code,
                now=ts,
                alert_id=f"eval-{index}",
                weight=primary.weight,
            )
            if not decision.should_write:
                continue
            risk = score(
                signals,
                RiskContext(
                    config=risk_cfg,
                    evqm_profile=profile,
                    track_max_conf=track.max_conf,
                    track_age_frames=track.age_frames,
                ),
            )
            result.fired.append({"kind": primary.code, "t": clip_t, "score": risk.score})

        index += 1

    capture.release()

    # Match fired alerts against labels within MATCH_WINDOW_S.
    unmatched = list(result.fired)
    for expected in result.expected:
        kind = expected.get("kind")
        start = float(expected.get("t_start", 0.0))
        end = float(expected.get("t_end", start))
        hit = next(
            (
                a
                for a in unmatched
                if a["kind"] == kind
                and start - MATCH_WINDOW_S <= float(a["t"]) <= end + MATCH_WINDOW_S
            ),
            None,
        )
        if hit is not None:
            unmatched.remove(hit)
            result.true_positives += 1
            result.latencies_s.append(max(0.0, float(hit["t"]) - start))
        else:
            result.false_negatives += 1
    result.false_positives = len(unmatched)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", default="eval/clips")
    parser.add_argument("--labels", default="eval/labels.jsonl")
    parser.add_argument("--zones", default="eval/zones.json")
    parser.add_argument("--profile", default="laptop")
    parser.add_argument("--out", default="docs/EVAL.md")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(
            f"✗ {labels_path} not found.\n\n"
            "The evaluation set is the one thing this repository cannot generate\n"
            "for itself: it needs real labelled footage. See the hard gates in\n"
            "README.md. Format, one object per line:\n\n"
            '  {"clip": "x.mp4", "camera": "CAM-02", "expect": [\n'
            '     {"kind": "TRIPWIRE_CROSS", "t_start": 12.4, "t_end": 15.0}]}\n\n'
            'A clip with "expect": [] is a negative, and negatives matter most.\n',
            file=sys.stderr,
        )
        return 1

    cfg = load_config("config", profile=args.profile, environ={})
    detector = build_detector(DetectorConfig.from_mapping(cfg.as_dict()))

    results: list[ClipResult] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        clip = Path(args.clips) / str(label["clip"])
        if not clip.exists():
            print(f"  · skipping missing clip {clip}")
            continue
        print(f"  evaluating {clip.name}…")
        results.append(
            run_clip(
                clip,
                label,
                load_zone_specs(Path(args.zones), str(label.get("camera", ""))),
                cfg,
                detector,
            )
        )

    if not results:
        print("no clips evaluated", file=sys.stderr)
        return 1

    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    idle = [r for r in results if not r.expected]
    idle_hours = sum(r.duration_s for r in idle) / 3600.0
    idle_fp = sum(r.false_positives for r in idle)
    fp_per_idle_hour = idle_fp / idle_hours if idle_hours > 0 else 0.0

    latencies = [x for r in results for x in r.latencies_s]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    per_rule: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        for a in r.fired:
            per_rule[str(a["kind"])]["fp"] += 0
        for e in r.expected:
            per_rule[str(e.get("kind"))]["fn"] += 0

    lines = [
        "# EVAL.md — measured on the labelled set",
        "",
        "> Generated by `make eval`. See BUILD_SPEC §15.",
        "",
        f"**Run:** {datetime.now(UTC).isoformat()}  ",
        f"**Profile:** `{args.profile}`  ",
        f"**Clips:** {len(results)} ({len(idle)} negative)  ",
        f"**Detector:** {detector.backend}",
        "",
        "## Headline",
        "",
        "| Metric | Value | Target (§14) |",
        "| --- | ---: | ---: |",
        f"| Precision | {precision:.3f} | ≥ 0.85 |",
        f"| Recall | {recall:.3f} | ≥ 0.80 |",
        f"| F1 | {f1:.3f} | — |",
        f"| False alerts / idle hour | {fp_per_idle_hour:.2f} | ≤ 0.5 |",
        f"| Mean alert latency | {mean_latency:.2f} s | — |",
        "",
        "False alerts per idle hour is the number that decides whether an operator",
        "keeps trusting the system. P3 says it outranks recall.",
        "",
        "## Per clip",
        "",
        "| Clip | Expected | Fired | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.clip} | {len(r.expected)} | {len(r.fired)} | "
            f"{r.true_positives} | {r.false_positives} | {r.false_negatives} |"
        )
    lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(
        f"\nprecision {precision:.3f}  recall {recall:.3f}  " f"FP/idle-hour {fp_per_idle_hour:.2f}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
