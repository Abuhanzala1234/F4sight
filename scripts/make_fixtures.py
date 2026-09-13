#!/usr/bin/env python3
"""Generate fixture videos (`make fixtures`).

Integration tests and the demo run against fixture RTSP streams served by
MediaMTX from local MP4s — never against a live camera (CLAUDE.md/Testing).

If you have real footage, drop it in ``infra/fixtures/`` with the expected
filenames and this script leaves it alone. Otherwise it synthesises clips with
OpenCV: a figure walking across a tripwire, the same at night, and a vehicle
arriving at a checkpoint. Synthetic footage is not a substitute for real
footage in evaluation, but it is enough to prove the pipeline end to end and it
needs no dataset licence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FIXTURES = Path("infra/fixtures")
W, H, FPS, SECONDS = 1280, 720, 25, 20


def _writer(path: Path):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


def _scene(night: bool):
    import numpy as np

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    ground, sky = (28, 32, 26), (46, 40, 34)
    if not night:
        ground, sky = (74, 96, 78), (150, 140, 120)
    frame[: int(H * 0.45)] = sky
    frame[int(H * 0.45) :] = ground
    return frame


def _draw_person(frame, x: int, y: int, night: bool) -> None:
    import cv2

    colour = (58, 58, 66) if night else (36, 38, 48)
    cv2.rectangle(frame, (x - 13, y - 78), (x + 13, y - 26), colour, -1)  # torso
    cv2.circle(frame, (x, y - 92), 13, colour, -1)  # head
    cv2.rectangle(frame, (x - 11, y - 26), (x - 3, y), colour, -1)  # legs
    cv2.rectangle(frame, (x + 3, y - 26), (x + 11, y), colour, -1)


def _draw_vehicle(frame, x: int, y: int) -> None:
    import cv2

    cv2.rectangle(frame, (x - 90, y - 62), (x + 90, y - 14), (72, 66, 58), -1)
    cv2.rectangle(frame, (x - 58, y - 96), (x + 46, y - 62), (82, 76, 68), -1)
    cv2.circle(frame, (x - 54, y - 8), 17, (22, 22, 24), -1)
    cv2.circle(frame, (x + 54, y - 8), 17, (22, 22, 24), -1)
    # A plate-sized light rectangle so the ANPR path has something to find.
    cv2.rectangle(frame, (x - 34, y - 40), (x + 34, y - 20), (226, 226, 220), -1)
    cv2.putText(
        frame,
        "HR26DA1234",
        (x - 31, y - 25),
        __import__("cv2").FONT_HERSHEY_SIMPLEX,
        0.42,
        (18, 18, 18),
        1,
    )


def _noise(frame, amount: int):
    import numpy as np

    if amount <= 0:
        return frame
    grain = np.random.default_rng(0).normal(0, amount, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + grain, 0, 255).astype("uint8")


def build_intrusion(path: Path, night: bool = False) -> None:
    import cv2

    out = _writer(path)
    total = FPS * SECONDS
    for i in range(total):
        frame = _scene(night).copy()
        # The tripwire, drawn faintly so the demo shows what the rule sees.
        cv2.line(frame, (W // 2, 0), (W // 2, H), (90, 90, 96), 1)
        x = int(W * 0.12 + (W * 0.76) * (i / total))
        _draw_person(frame, x, int(H * 0.78), night)
        out.write(_noise(frame, 9 if night else 3))
    out.release()
    print(f"  ✓ {path}  ({SECONDS}s, person crosses the wire at ~{SECONDS // 2}s)")


def build_vehicle(path: Path) -> None:
    out = _writer(path)
    total = FPS * SECONDS
    for i in range(total):
        frame = _scene(False).copy()
        progress = i / total
        # Drive in, stop in the bay, drive out — so plate voting gets several
        # frames of a stationary plate, which is the realistic case.
        if progress < 0.35:
            x = int(-200 + (W * 0.5 + 200) * (progress / 0.35))
        elif progress < 0.7:
            x = int(W * 0.5)
        else:
            x = int(W * 0.5 + (W * 0.7) * ((progress - 0.7) / 0.3))
        _draw_vehicle(frame, x, int(H * 0.82))
        out.write(_noise(frame, 3))
    out.release()
    print(f"  ✓ {path}  ({SECONDS}s, vehicle halts in the bay ~7-14s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="regenerate even if present"
    )
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print(
            "OpenCV is required to synthesise fixtures:\n"
            "  pip install opencv-python-headless\n"
            "Alternatively drop real MP4s into infra/fixtures/ with these names:\n"
            "  intrusion_day.mp4  night_movement.mp4  vehicle_gate.mp4",
            file=sys.stderr,
        )
        return 1

    print("Generating fixture clips in infra/fixtures/\n")
    targets = [
        (FIXTURES / "intrusion_day.mp4", lambda p: build_intrusion(p, night=False)),
        (FIXTURES / "night_movement.mp4", lambda p: build_intrusion(p, night=True)),
        (FIXTURES / "vehicle_gate.mp4", build_vehicle),
    ]
    for path, build in targets:
        if path.exists() and not args.force:
            print(f"  · {path} already exists (use --force to regenerate)")
            continue
        build(path)

    print(
        "\nMediaMTX republishes these as RTSP on `make up`:\n"
        "  rtsp://localhost:8554/fixture-intrusion\n"
        "  rtsp://localhost:8554/fixture-night\n"
        "  rtsp://localhost:8554/fixture-vehicle\n"
        "\nSynthetic footage proves the pipeline. It is NOT a substitute for real\n"
        "footage in `make eval` — see the hard gates in README.md.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
