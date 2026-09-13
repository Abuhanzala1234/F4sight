#!/usr/bin/env python3
"""Download and export every free model (`make models`). See docs/MODELS.md.

Total ~166 MB, once. After this the box can be offline forever (P9).

Writes ``models/MANIFEST.json`` with a SHA-256 per file. The worker verifies
that manifest at startup and refuses to run on a mismatch — which catches both
supply-chain tampering and, far more often, a half-downloaded file that would
otherwise fail mysteriously at frame 4000.

Air-gapped install: run this on a connected machine, copy ``models/`` across.
The manifest makes the copy verifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path("models")
MANIFEST = MODELS_DIR / "MANIFEST.json"


@dataclass(frozen=True)
class Artefact:
    key: str
    path: str
    url: str
    licence: str
    purpose: str
    # Ultralytics ships .pt; we export to ONNX so the runtime is onnxruntime and
    # there is no torch dependency at inference time.
    export_onnx: bool = False
    optional: bool = False
    notes: str = ""


ARTEFACTS: tuple[Artefact, ...] = (
    Artefact(
        key="detector.yolo11n",
        path="models/detect/yolo11n.onnx",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        licence="AGPL-3.0",
        purpose="person / vehicle / animal / bag detection (laptop profile)",
        export_onnx=True,
    ),
    Artefact(
        key="detector.yolo11s",
        path="models/detect/yolo11s.onnx",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        licence="AGPL-3.0",
        purpose="the same, bop profile",
        export_onnx=True,
        optional=True,
    ),
    Artefact(
        key="face.detect",
        path="models/face/scrfd_500m.onnx",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        licence="MIT (code), free weights",
        purpose="face detection — OPT-IN, disabled by default (P6)",
        optional=True,
        notes="zip; extract det_500m.onnx",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  ↓ {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open(
            "wb"
        ) as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  ✗ download failed: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    # Rename only after a complete download, so a half-file never looks valid.
    tmp.rename(dest)
    return True


def export_to_onnx(pt_path: Path, onnx_path: Path) -> bool:
    """Export an Ultralytics .pt to ONNX opset 12.

    Needs `ultralytics` installed, which pulls in torch. That is a build-time
    dependency only — the worker runs on onnxruntime and never imports torch.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "  ✗ the `ultralytics` package is needed to export ONNX.\n"
            "    pip install ultralytics   (build-time only; the worker does not need it)",
            file=sys.stderr,
        )
        return False

    print(f"  → exporting {pt_path.name} to ONNX (opset 12)")
    model = YOLO(str(pt_path))
    produced = model.export(format="onnx", opset=12, simplify=True, dynamic=True)
    Path(produced).replace(onnx_path)
    return True


def fetch(artefact: Artefact, force: bool) -> tuple[bool, str]:
    dest = Path(artefact.path)
    if dest.exists() and not force:
        return True, "present"

    if artefact.export_onnx:
        pt = dest.with_suffix(".pt")
        if (not pt.exists() or force) and not download(artefact.url, pt):
            return False, "download failed"
        if not export_to_onnx(pt, dest):
            return False, "export failed"
        pt.unlink(missing_ok=True)
        return True, "downloaded + exported"

    if not download(artefact.url, dest):
        return False, "download failed"
    return True, "downloaded"


def build_manifest() -> dict[str, object]:
    files = []
    for artefact in ARTEFACTS:
        path = Path(artefact.path)
        if path.exists():
            files.append(
                {
                    "key": artefact.key,
                    "path": artefact.path,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "licence": artefact.licence,
                    "purpose": artefact.purpose,
                    "source": artefact.url,
                }
            )
    return {
        "schema": "drishti.models/v1",
        "note": (
            "Every model here is free to download and run locally, with no account, "
            "no API key and no metered call. See docs/MODELS.md."
        ),
        "files": files,
    }


def verify_only() -> int:
    if not MANIFEST.exists():
        print(f"✗ {MANIFEST} not found. Run `make models` first.", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    failures = 0
    for entry in manifest.get("files", []):
        path = Path(entry["path"])
        if not path.exists():
            print(f"✗ missing: {path}", file=sys.stderr)
            failures += 1
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            print(
                f"✗ hash mismatch: {path}\n    expected {entry['sha256']}\n"
                f"    actual   {actual}",
                file=sys.stderr,
            )
            failures += 1
        else:
            print(f"✓ {path}  {entry['bytes'] / 1e6:.1f} MB  {entry['licence']}")
    if failures:
        print(
            f"\n{failures} problem(s). Re-download with `make models FORCE=1`.",
            file=sys.stderr,
        )
        return 1
    print(f"\n✓ all {len(manifest.get('files', []))} model files verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="check hashes, download nothing"
    )
    parser.add_argument("--skip-optional", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        return verify_only()

    MODELS_DIR.mkdir(exist_ok=True)
    print("DRISHTI-BOP model fetch — every file below is free. See docs/MODELS.md.\n")

    failed_required = []
    for artefact in ARTEFACTS:
        if artefact.optional and args.skip_optional:
            continue
        print(f"{artefact.key}  ({artefact.licence})")
        ok, status = fetch(artefact, args.force)
        print(f"  {'✓' if ok else '✗'} {status}")
        if not ok and not artefact.optional:
            failed_required.append(artefact.key)
        print()

    MANIFEST.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    print(f"wrote {MANIFEST}")

    print("\n" + "─" * 62)
    print("  LICENCE SUMMARY — nobody can say they were not told")
    print("─" * 62)
    for artefact in ARTEFACTS:
        if Path(artefact.path).exists():
            print(f"  {artefact.licence:<26} {artefact.key}")
    print(
        "\n  Ultralytics YOLO is AGPL-3.0. If that is unacceptable to the\n"
        "  deploying organisation, set detector.backend=rtdetr in\n"
        "  config/detector.yaml for the Apache-2.0 path (docs/MODELS.md §2).\n"
    )

    if failed_required:
        print(
            f"✗ required models failed: {', '.join(failed_required)}", file=sys.stderr
        )
        print("  The worker can still run with detector.backend=mock.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
