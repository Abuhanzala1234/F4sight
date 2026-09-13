#!/usr/bin/env python3
"""`make demo` — the demo-day command. Cold start, offline, everything.

Starts the worker, the API and the anchor service together, waits for each to
report ready, and prints the URLs. Ctrl-C stops all three cleanly.

Deliberately boring. Demo day is not the time to discover that a start-up
script has opinions.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Windows' console defaults to the system codepage (cp1252), not UTF-8, and
# this script prints ✓/✗/→ in its progress output -- a plain print() of
# any of them raises UnicodeEncodeError before the actual work even starts.
# reconfigure() is a no-op everywhere already UTF-8; errors="replace" means a
# console that truly cannot show a glyph gets a "?" instead of a crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
_VENV_CANDIDATES = (
    ROOT / ".venv" / "bin" / "python",  # POSIX venv layout
    ROOT / ".venv" / "Scripts" / "python.exe",  # Windows venv layout
)
VENV_PY = next((p for p in _VENV_CANDIDATES if p.exists()), None)
PY = str(VENV_PY if VENV_PY is not None else sys.executable)

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"


class Service:
    def __init__(self, name: str, argv: list[str], ready_url: str | None = None) -> None:
        self.name = name
        self.argv = argv
        self.ready_url = ready_url
        self.process: subprocess.Popen[bytes] | None = None

    def start(self, env: dict[str, str]) -> None:
        print(f"  starting {self.name}…")
        self.process = subprocess.Popen(self.argv, cwd=ROOT, env=env)

    def wait_ready(self, timeout_s: float = 60.0) -> bool:
        if self.ready_url is None:
            time.sleep(2.0)
            return self.alive
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.alive:
                return False
            try:
                with urllib.request.urlopen(self.ready_url, timeout=2) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(0.5)
        return False

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if not self.alive or self.process is None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            print(f"  {self.name} did not stop; killing it")
            self.process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.getenv("DRISHTI_PROFILE", "laptop"))
    parser.add_argument("--site", default=os.getenv("DRISHTI_SITE", "BOP-03"))
    parser.add_argument("--no-worker", action="store_true", help="API and anchor only")
    args = parser.parse_args()

    env = {
        **os.environ,
        # os.pathsep, not a literal ":" -- Windows needs ";", and a worker or
        # API subprocess launched with the wrong separator silently gets an
        # empty PYTHONPATH (the whole string is treated as one nonexistent
        # path) and fails at its very first `import drishti_api`/
        # `import drishti_worker`, before either service prints a line.
        "PYTHONPATH": os.pathsep.join([str(ROOT / "worker" / "src"), str(ROOT / "api" / "src")]),
    }

    print(f"\n{BOLD}DRISHTI-BOP — demo{RESET}")
    print(f"{DIM}SIH 2026 · PS 26187 · Team SW-73 (ByteForge){RESET}")
    print(f"{DIM}profile={args.profile} site={args.site}{RESET}\n")

    services = [
        Service(
            "api",
            [
                PY,
                "-m",
                "uvicorn",
                "drishti_api.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            ready_url="http://localhost:8000/api/v1/health/live",
        ),
        Service(
            "anchor",
            [PY, "-m", "drishti_worker.anchor_service", "--profile", args.profile],
        ),
    ]
    if not args.no_worker:
        services.append(
            Service(
                "worker",
                [
                    PY,
                    "-m",
                    "drishti_worker",
                    "--profile",
                    args.profile,
                    "--site",
                    args.site,
                ],
            )
        )

    started: list[Service] = []
    try:
        for service in services:
            service.start(env)
            started.append(service)
            ok = service.wait_ready()
            print(f"  {GREEN + '✓' if ok else RED + '✗'} {service.name}{RESET}")
            if not ok and service.name == "api":
                print(f"{RED}the API did not come up; aborting{RESET}", file=sys.stderr)
                raise SystemExit(1)

        print(f"\n{BOLD}  ready{RESET}")
        print(f"    dashboard   {BOLD}http://localhost:8000/{RESET}   (built assets)")
        print("    api docs    http://localhost:8000/api/docs")
        print("    health      http://localhost:8000/api/v1/health")
        print("    hls         http://localhost:8888/fixture-intrusion/index.m3u8")
        print("    minio       http://localhost:9001")
        print(f"\n{DIM}  Ctrl-C to stop everything.{RESET}\n")

        while all(s.alive for s in started):
            time.sleep(1.0)

        for service in started:
            if not service.alive:
                print(f"{RED}{service.name} exited unexpectedly{RESET}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\n  stopping…")
        return 0
    finally:
        for service in reversed(started):
            service.stop()
        print("  all services stopped")


if __name__ == "__main__":
    raise SystemExit(main())
