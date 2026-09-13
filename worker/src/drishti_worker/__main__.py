"""Worker entrypoint — `make worker` (BUILD_SPEC §7.14, §16 Phase 2+).

Startup order matters and is deliberate:

1. load + version config (every alert must be attributable to a ruleset)
2. check the clock (§13 — evidence timestamps must be trustworthy)
3. verify the model manifest (a half-downloaded weight file fails loudly here,
   not silently at frame 4000)
4. build and **warm up** the detector (blocker #3) — nothing reports ready first
5. load cameras and zones
6. start the pipeline

The process reports ready only after step 4. A dashboard that says "starting"
for twenty seconds is honest; one that says "live" and shows nothing is not.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime
from typing import Any

from .alerting import AlertAssembler
from .anpr import AnprConfig
from .config import AppConfig, load_config
from .detect import DetectorConfig, build_detector
from .enhance import EnhanceConfig
from .evqm import EVQMConfig
from .ingest import IngestConfig
from .logsetup import configure_logging
from .pipeline import CameraWorker, Pipeline
from .risk import RiskConfig
from .rules import DebounceConfig, RuleConfig
from .sinks import (
    FanoutSink,
    LiveTrackPublisher,
    MinioSink,
    NullSink,
    PostgresSink,
    RedisSink,
    SpoolSink,
)
from .track import TrackerConfig
from .types import Calibration, CameraRuntime, ZoneKind, ZoneRuntime
from .watchlist import WatchlistCache

logger = logging.getLogger(__name__)


def check_clock(max_skew_s: float) -> None:
    """Refuse to start on a badly skewed clock (§13).

    Evidence timestamps are part of the hashed document and are what an
    investigator correlates against radio logs. A worker two minutes off
    produces evidence that looks fabricated.
    """
    try:
        import socket
        import struct

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(2.0)
        client.sendto(b"\x1b" + 47 * b"\0", ("pool.ntp.org", 123))
        data, _ = client.recvfrom(1024)
        ntp_time = struct.unpack("!12I", data)[10] - 2208988800
        skew = abs(time.time() - ntp_time)
        if skew > max_skew_s:
            raise SystemExit(
                f"system clock is {skew:.1f}s off NTP (limit {max_skew_s}s). "
                f"Evidence timestamps would not be trustworthy. Fix the clock first."
            )
        logger.info("clock check passed (skew %.2fs)", skew)
    except (OSError, TimeoutError, struct.error):
        # P9: offline is the normal case at a BOP. No uplink is not a failure.
        logger.warning("NTP unreachable; skipping the clock check (offline operation assumed)")


def verify_models(cfg: AppConfig) -> None:
    """Fail loudly on a bad or missing model manifest (docs/MODELS.md §4)."""
    if not cfg.get("runtime.verify_model_manifest", True):
        return
    if cfg.get("detector.backend", "onnx") == "mock":
        logger.info("detector backend is 'mock'; skipping model manifest verification")
        return

    import hashlib
    import json
    from pathlib import Path

    manifest_path = Path(cfg.get("runtime.models_dir", "models")) / "MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} not found. Run `make models` to download the free "
            f"model weights (~166 MB, one time). See docs/MODELS.md."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("files", []):
        path = Path(entry["path"])
        if not path.exists():
            raise SystemExit(f"model file missing: {path}. Run `make models`.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(
                f"model file {path} does not match the manifest.\n"
                f"  expected {entry['sha256']}\n  actual   {digest}\n"
                f"Re-download with `make models FORCE=1`."
            )
    logger.info("model manifest verified (%d files)", len(manifest.get("files", [])))


def load_cameras(cfg: AppConfig) -> list[tuple[CameraRuntime, str, list[ZoneRuntime]]]:
    """Load cameras and zones from Postgres, or fall back to fixtures.

    The fallback is what makes `make demo` work on a machine with no database
    yet — the fixture streams are described in ``config/`` so the worker has
    something to analyse from a cold start.
    """
    dsn = _dsn()
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3) as conn:
            rows = conn.execute("""
                SELECT c.id::text, c.code, c.site_id::text, s.code, s.timezone,
                       c.resolution_w, c.resolution_h, c.analytics_fps,
                       c.rtsp_url, c.mediamtx_path, c.calibration
                FROM camera c JOIN site s ON s.id = c.site_id
                WHERE c.enabled AND NOT c.is_recording_only
                ORDER BY s.code, c.code
                """).fetchall()

            out: list[tuple[CameraRuntime, str, list[ZoneRuntime]]] = []
            for r in rows:
                camera = CameraRuntime(
                    camera_id=r[0],
                    code=r[1],
                    site_id=r[2],
                    site_code=r[3],
                    timezone=r[4] or "UTC",
                    width=r[5],
                    height=r[6],
                    analytics_fps=float(r[7]),
                    calibration=_calibration(r[10]),
                )
                zone_rows = conn.execute(
                    """
                    SELECT id::text, name, kind, polygon, direction, classes,
                           schedule, severity_base, enabled
                    FROM zone WHERE camera_id = %s AND enabled
                    """,
                    (r[0],),
                ).fetchall()
                zones = [_zone(z, camera.width, camera.height) for z in zone_rows]
                source = r[8] or f"rtsp://{os.getenv('MEDIAMTX_HOST','localhost')}:8554/{r[9]}"
                out.append((camera, source, zones))

            if out:
                logger.info("loaded %d cameras from the database", len(out))
                return out
    except Exception as exc:
        logger.warning(
            "could not load cameras from the database (%s); "
            "falling back to fixture streams from config",
            exc,
        )

    return _fixture_cameras(cfg)


def _fixture_cameras(
    cfg: AppConfig,
) -> list[tuple[CameraRuntime, str, list[ZoneRuntime]]]:
    host = os.getenv("MEDIAMTX_HOST", "localhost")
    port = os.getenv("RTSP_PORT", "8554")
    site_code = cfg.site or "BOP-03"
    fps = float(cfg.get("ingest.analytics_fps", 6.0))

    camera = CameraRuntime(
        camera_id="fixture-cam-01",
        code="CAM-01",
        site_id="fixture-site",
        site_code=site_code,
        timezone=cfg.get("site.timezone", "Asia/Kolkata"),
        width=1280,
        height=720,
        analytics_fps=fps,
    )
    # A tripwire across the middle and a restricted area on the right. Both in
    # pixel space for a 1280x720 fixture.
    zones = [
        ZoneRuntime(
            zone_id="fixture-wire",
            name="Perimeter line",
            kind=ZoneKind.TRIPWIRE,
            polygon=((640.0, 0.0), (640.0, 720.0)),
            direction="both",
            classes=("person", "vehicle"),
            severity_base=4,
        ),
        ZoneRuntime(
            zone_id="fixture-area",
            name="Restricted zone",
            kind=ZoneKind.AREA,
            polygon=((900.0, 200.0), (1280.0, 200.0), (1280.0, 720.0), (900.0, 720.0)),
            classes=("person",),
            severity_base=3,
        ),
    ]
    source = f"rtsp://{host}:{port}/fixture-intrusion"
    logger.info("using fixture camera %s -> %s", camera.code, source)
    return [(camera, source, zones)]


def _zone(row: Any, width: int, height: int) -> ZoneRuntime:
    from .geometry import denormalise

    polygon = [(float(p[0]), float(p[1])) for p in (row[3] or [])]
    return ZoneRuntime(
        zone_id=row[0],
        name=row[1],
        kind=ZoneKind(row[2]),
        polygon=tuple(denormalise(polygon, width, height)),
        direction=row[4],
        classes=tuple(row[5] or ()),
        schedule=row[6],
        severity_base=int(row[7] or 3),
        enabled=bool(row[8]),
    )


def _calibration(raw: Any) -> Calibration | None:
    if not raw:
        return None
    return Calibration(
        homography=tuple(raw["homography"]) if raw.get("homography") else None,
        px_per_m_at_y=tuple(tuple(x) for x in raw.get("px_per_m_at_y", ())),
    )


def _dsn() -> str:
    return (
        f"postgresql://{os.getenv('DB_USER', 'drishti')}:"
        f"{os.getenv('DB_PASSWORD', 'drishti_dev')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'drishti')}"
    )


def build_sinks(cfg: AppConfig) -> tuple[FanoutSink, MinioSink | None]:
    sinks: list[Any] = []
    minio: MinioSink | None = None

    if cfg.get("sinks.postgres", True):
        sinks.append(PostgresSink(_dsn()))
    if cfg.get("sinks.redis", True):
        sinks.append(RedisSink(os.getenv("REDIS_URL", "redis://localhost:6379/0")))
    if cfg.get("sinks.minio", True):
        minio = MinioSink(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "drishti"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "drishti_dev_secret"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )
    if not sinks:
        sinks.append(NullSink())

    spool = (
        SpoolSink(cfg.get("runtime.spool_dir", "spool"))
        if cfg.get("sinks.spool_on_failure", True)
        else None
    )
    return (
        FanoutSink(sinks=sinks, spool=spool, fail_soft=cfg.get("sinks.fail_soft", True)),
        minio,
    )


def build_anpr(cfg: AppConfig) -> tuple[AnprConfig, Any, bytes]:
    """§7.9. Returns the config, a shared reader (or None if disabled/unbuilt),
    and the plate HMAC key.

    Fails soft by design: a missing or too-short key does not stop the
    worker, it just means settled plates are never checked against the
    watchlist (WatchlistPlateRule.evaluate short-circuits on ``plate_hit is
    None``) -- the same posture as a missing model weight file for a
    non-critical stage.
    """
    anpr_cfg = AnprConfig.from_mapping(cfg.as_dict())
    if not anpr_cfg.enabled:
        return anpr_cfg, None, b""

    from .anpr import build_reader

    reader = build_reader(cfg.as_dict())
    key = os.getenv(anpr_cfg.hmac_key_env, "").encode()
    if not key:
        logger.warning(
            "%s is not set; ANPR will read and vote on plates but cannot check "
            "them against the watchlist",
            anpr_cfg.hmac_key_env,
        )
    elif len(key) < 16:
        logger.warning(
            "%s is only %d bytes (need >= 16); ANPR watchlist checks are disabled",
            anpr_cfg.hmac_key_env,
            len(key),
        )
        key = b""
    return anpr_cfg, reader, key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="drishti-worker", description="DRISHTI-BOP analytics worker"
    )
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile", default=None, help="laptop | bop | edge")
    parser.add_argument("--site", default=None, help="site code, e.g. BOP-03")
    parser.add_argument("--source", default=None, help="override: one RTSP URL or MP4 path")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, profile=args.profile, site=args.site)
    configure_logging(
        args.log_level or cfg.get("logging.level", "INFO"),
        cfg.get("logging.format", "console"),
    )

    logger.info("DRISHTI-BOP worker starting — %s", cfg.summary())
    logger.info("config sources: %s", ", ".join(cfg.sources))

    check_clock(float(cfg.get("runtime.max_clock_skew_s", 2.0)))
    verify_models(cfg)

    detector_cfg = DetectorConfig.from_mapping(cfg.as_dict())
    detector = build_detector(detector_cfg)  # warms up before returning (blocker #3)

    cameras = load_cameras(cfg)
    if args.source:
        camera, _, zones = cameras[0]
        cameras = [(camera, args.source, zones)]

    fanout, minio = build_sinks(cfg)
    assembler = AlertAssembler(
        sinks=fanout,
        minio=minio,
        config_version=cfg.version,
        spec_version=cfg.get("meta.spec_version", "1.0.0"),
        worker_version=__import__("drishti_worker").__version__,
        snapshot_quality=int(cfg.get("evidence.snapshot_quality", 92)),
        clip_pre_roll_s=float(cfg.get("evidence.clip_pre_roll_s", 5.0)),
        clip_post_roll_s=float(cfg.get("evidence.clip_post_roll_s", 5.0)),
        clip_fps=int(cfg.get("evidence.clip_fps", 8)),
    )

    def on_alert(**kwargs: Any) -> None:
        record = assembler.build(**kwargs)
        assembler.emit(record)

    live_tracks = LiveTrackPublisher(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    pipeline = Pipeline(
        detector,
        detector_cfg,
        frame_queue_size=int(cfg.get("ingest.frame_queue_size", 4)),
        stats_interval_s=float(cfg.get("pipeline.stats_interval_s", 10)),
    )

    anpr_cfg, anpr_reader, plate_hmac_key = build_anpr(cfg)
    watchlist = WatchlistCache()
    if anpr_reader is not None:
        # Best-effort from the start: a DB that is not up yet just means no
        # watchlist hits until the first successful refresh (P9). ANPR keeps
        # reading and voting on plates regardless -- only the watchlist check
        # depends on this.
        watchlist.start_refresh_thread(
            _dsn(), interval_s=float(cfg.get("anpr.watchlist_refresh_s", 60.0))
        )

    for camera, source, zones in cameras:
        pipeline.add_worker(
            CameraWorker(
                camera=camera,
                source=source,
                zones=zones,
                ingest_cfg=IngestConfig.from_mapping(cfg.as_dict(), source),
                evqm_cfg=EVQMConfig.from_mapping(cfg.as_dict()),
                enhance_cfg=EnhanceConfig.from_mapping(cfg.as_dict()),
                tracker_cfg=TrackerConfig.from_mapping(cfg.as_dict()),
                rule_cfg=RuleConfig.from_mapping(cfg.as_dict()),
                risk_cfg=RiskConfig.from_mapping(cfg.as_dict()),
                debounce_cfg=DebounceConfig.from_mapping(cfg.as_dict()),
                frame_queue=pipeline.frame_queue,
                on_alert=on_alert,
                on_tracks=live_tracks.publish,
                clip_pre_roll_s=float(cfg.get("evidence.clip_pre_roll_s", 5.0)),
                anpr_cfg=anpr_cfg,
                anpr_reader=anpr_reader,
                watchlist=watchlist,
                plate_hmac_key=plate_hmac_key,
            )
        )

    stopping = {"flag": False}

    def shutdown(*_: Any) -> None:
        if stopping["flag"]:
            return
        stopping["flag"] = True
        logger.info("shutdown requested; stopping pipeline")
        pipeline.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    pipeline.start()
    logger.info(
        "READY — %d cameras, detector=%s, config_version=%s",
        len(cameras),
        detector.backend,
        cfg.version[:12],
    )
    print(f"[{datetime.now(UTC).isoformat()}] drishti-worker READY", file=sys.stderr)

    try:
        while not stopping["flag"]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
