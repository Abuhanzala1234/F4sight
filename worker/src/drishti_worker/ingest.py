"""RTSP ingest with watchdog and reconnect (BUILD_SPEC §7.1). **[DEMO-CRITICAL]**

Blocker #2, quoting CLAUDE.md:

    RTSP stalls. OpenCV's VideoCapture hangs silently on dropped RTSP. The
    watchdog + reconnect in §7.1 is mandatory from Phase 2.

The failure is worse than it sounds. When an RTSP source dies, ``read()`` may:

* return ``(False, None)`` forever — detectable, annoying;
* return ``(True, <the same stale frame>)`` — undetectable from the return value;
* **block inside FFmpeg for 30+ seconds** — undetectable AND unkillable from the
  calling thread.

The third case is why the watchdog is a *separate thread*. A wedged reader
cannot rescue itself, by definition. The watchdog watches a monotonic timestamp
that the reader stamps before each frame, and when it goes stale the watchdog
calls ``capture.release()`` from the outside — which is what forces the blocked
``read()`` to return so the reader can reconnect.

``FileReader`` implements the same interface over a local MP4, so tests and the
offline demo never need a network or a camera.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .types import Frame, StreamState

logger = logging.getLogger(__name__)

__all__ = ["FileReader", "IngestConfig", "IngestStats", "RtspReader"]

OnFrame = Callable[[Frame], None]
OnState = Callable[[str, StreamState, "str | None"], None]


@dataclass(frozen=True, slots=True)
class IngestConfig:
    rtsp_url: str = ""
    analytics_fps: float = 6.0
    read_timeout_s: float = 5.0
    reconnect_backoff_s: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)
    max_consecutive_failures: int = 0  # 0 = retry forever (P9: offline is normal)
    transport: str = "tcp"
    buffer_size: int = 1
    watchdog_interval_s: float = 1.0
    open_timeout_us: int = 5_000_000

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any], rtsp_url: str = "") -> IngestConfig:
        block = dict(cfg.get("ingest", cfg))
        backoff = block.get("reconnect_backoff_s", (1, 2, 4, 8, 15, 30))
        return cls(
            rtsp_url=rtsp_url,
            analytics_fps=float(block.get("analytics_fps", 6.0)),
            read_timeout_s=float(block.get("read_timeout_s", 5.0)),
            reconnect_backoff_s=tuple(float(x) for x in backoff),
            max_consecutive_failures=int(block.get("max_consecutive_failures", 0)),
            transport=str(block.get("transport", "tcp")),
            buffer_size=int(block.get("buffer_size", 1)),
            watchdog_interval_s=float(block.get("watchdog_interval_s", 1.0)),
        )


@dataclass
class IngestStats:
    frames_read: int = 0
    frames_emitted: int = 0
    frames_dropped_fps: int = 0
    reconnects: int = 0
    stalls: int = 0
    consumer_errors: int = 0
    last_error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    last_frame_at: float | None = None

    @property
    def fps_in(self) -> float:
        elapsed = time.monotonic() - self.started_at
        return self.frames_read / elapsed if elapsed > 0 else 0.0

    @property
    def fps_emitted(self) -> float:
        elapsed = time.monotonic() - self.started_at
        return self.frames_emitted / elapsed if elapsed > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_read": self.frames_read,
            "frames_emitted": self.frames_emitted,
            "frames_dropped_fps": self.frames_dropped_fps,
            "reconnects": self.reconnects,
            "stalls": self.stalls,
            "consumer_errors": self.consumer_errors,
            "fps_in": round(self.fps_in, 2),
            "fps_emitted": round(self.fps_emitted, 2),
            "last_error": self.last_error,
        }


class _BaseReader:
    """Shared lifecycle, frame pacing and state reporting."""

    def __init__(
        self,
        camera_id: str,
        cfg: IngestConfig,
        on_frame: OnFrame,
        on_state: OnState | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.cfg = cfg
        self._on_frame = on_frame
        self._on_state = on_state
        self._state = StreamState.CONNECTING
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = IngestStats()
        self._frame_id = 0
        self._last_emit_mono = 0.0
        self._pending_gap = 0

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError(f"reader for {self.camera_id} is already running")
        self._stop.clear()
        # Named threads: a stack dump at 2 a.m. must be readable (§7.14).
        self._thread = threading.Thread(
            target=self._run, name=f"reader-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.error(
                    "camera=%s reader thread did not stop within %.1fs; abandoning it",
                    self.camera_id,
                    timeout_s,
                )
        self._set_state(StreamState.FAILED if self._state != StreamState.LIVE else self._state)

    def _set_state(self, state: StreamState, error: str | None = None) -> None:
        if state == self._state and error is None:
            return
        previous, self._state = self._state, state
        logger.info(
            "camera=%s stream %s -> %s%s",
            self.camera_id,
            previous,
            state,
            f" ({error})" if error else "",
        )
        if self._on_state is not None:
            try:
                self._on_state(self.camera_id, state, error)
            except Exception:
                logger.exception("camera=%s state callback failed", self.camera_id)

    def _should_emit(self, now_mono: float) -> bool:
        """Frame dropping by TIMESTAMP, not by ``frame_id % n``.

        Variable-rate sources exist — a camera that drops to 8 fps in low light
        would, under modulo dropping, silently hand us 2 fps of analytics.
        """
        if self.cfg.analytics_fps <= 0:
            return True
        interval = 1.0 / self.cfg.analytics_fps
        if now_mono - self._last_emit_mono >= interval:
            self._last_emit_mono = now_mono
            return True
        return False

    def _emit(self, image: Any, width: int, height: int) -> None:
        """Hand a frame to the consumer. A consumer error never stalls decode."""
        self._frame_id += 1
        frame = Frame(
            camera_id=self.camera_id,
            frame_id=self._frame_id,
            ts_utc=datetime.now(UTC),
            image=image,
            width=width,
            height=height,
            seq_gap=self._pending_gap,
        )
        self._pending_gap = 0
        try:
            self._on_frame(frame)
            self.stats.frames_emitted += 1
        except Exception:
            # §7.1 rule 7: on_frame must never raise into the reader.
            self.stats.consumer_errors += 1
            logger.exception(
                "camera=%s frame consumer raised on frame_id=%d; dropping this frame",
                self.camera_id,
                frame.frame_id,
            )

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class RtspReader(_BaseReader):
    """One reader thread plus one watchdog thread per camera."""

    def __init__(
        self,
        camera_id: str,
        cfg: IngestConfig,
        on_frame: OnFrame,
        on_state: OnState | None = None,
    ) -> None:
        super().__init__(camera_id, cfg, on_frame, on_state)
        self._capture: Any = None
        self._capture_lock = threading.Lock()
        self._last_frame_mono = time.monotonic()
        self._watchdog: threading.Thread | None = None
        self._reconnect_requested = threading.Event()

    def start(self) -> None:
        super().start()
        self._watchdog = threading.Thread(
            target=self._watch, name=f"watchdog-{self.camera_id}", daemon=True
        )
        self._watchdog.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        super().stop(timeout_s)
        if self._watchdog is not None:
            self._watchdog.join(timeout=timeout_s)
        self._release()

    # -- reader ------------------------------------------------------------

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            if not self._open():
                attempt += 1
                if (
                    self.cfg.max_consecutive_failures
                    and attempt >= self.cfg.max_consecutive_failures
                ):
                    self._set_state(
                        StreamState.FAILED,
                        f"gave up after {attempt} consecutive failures",
                    )
                    return
                self._backoff(attempt)
                continue

            attempt = 0
            self._set_state(StreamState.LIVE)
            self._reconnect_requested.clear()
            self._pump()

            if self._stop.is_set():
                break
            self.stats.reconnects += 1
            self._set_state(StreamState.RECONNECTING)
            self._release()
            attempt += 1
            self._backoff(attempt)

        self._release()

    def _pump(self) -> None:
        """Read frames until the stream dies or the watchdog intervenes."""
        while not self._stop.is_set() and not self._reconnect_requested.is_set():
            with self._capture_lock:
                capture = self._capture
                if capture is None:
                    return
                ok, image = capture.read()

            # Stamp BEFORE delivering the frame (§7.1 rule 3). Stamping after
            # would let a slow consumer look like a stalled camera.
            self._last_frame_mono = time.monotonic()

            if not ok or image is None:
                self.stats.last_error = "read() returned no frame"
                logger.warning("camera=%s read() returned no frame; reconnecting", self.camera_id)
                return

            self.stats.frames_read += 1
            if self._should_emit(self._last_frame_mono):
                h, w = image.shape[:2]
                self._emit(image, w, h)
            else:
                self.stats.frames_dropped_fps += 1
                self._pending_gap += 1

    def _open(self) -> bool:
        import cv2

        self._set_state(StreamState.CONNECTING)
        # RTSP over TCP with a socket timeout. UDP loses frames silently, which
        # at a border post means losing the one frame that mattered.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.cfg.transport}"
            f"|stimeout;{self.cfg.open_timeout_us}"
            f"|max_delay;500000"
        )
        try:
            capture = cv2.VideoCapture(self.cfg.rtsp_url, cv2.CAP_FFMPEG)
            # Newest frame only. A backed-up buffer is latency we can never
            # recover, and stale frames produce alerts about the past.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.buffer_size)
        except Exception as exc:
            self.stats.last_error = str(exc)
            logger.exception("camera=%s failed to construct VideoCapture", self.camera_id)
            return False

        if not capture.isOpened():
            capture.release()
            self.stats.last_error = "VideoCapture.isOpened() is False"
            logger.warning(
                "camera=%s could not open %s",
                self.camera_id,
                _redact(self.cfg.rtsp_url),
            )
            return False

        with self._capture_lock:
            self._capture = capture
        self._last_frame_mono = time.monotonic()
        return True

    def _release(self) -> None:
        with self._capture_lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    logger.exception("camera=%s error releasing capture", self.camera_id)
                self._capture = None

    def _backoff(self, attempt: int) -> None:
        schedule = self.cfg.reconnect_backoff_s or (1.0,)
        delay = schedule[min(attempt - 1, len(schedule) - 1)]
        logger.warning(
            "camera=%s reconnect attempt %d in %.1fs (last error: %s)",
            self.camera_id,
            attempt,
            delay,
            self.stats.last_error,
        )
        self._stop.wait(delay)

    # -- watchdog ----------------------------------------------------------

    def _watch(self) -> None:
        """Separate thread. A wedged reader cannot rescue itself.

        Releasing the capture from HERE is the trick: it forces the blocked
        ``read()`` inside FFmpeg to return, which is the only reliable way out
        of a silent RTSP hang.
        """
        while not self._stop.wait(self.cfg.watchdog_interval_s):
            if self._state is not StreamState.LIVE:
                continue
            stale_for = time.monotonic() - self._last_frame_mono
            if stale_for <= self.cfg.read_timeout_s:
                continue

            self.stats.stalls += 1
            self.stats.last_error = f"no frame for {stale_for:.1f}s"
            self._set_state(StreamState.STALLED, self.stats.last_error)
            logger.warning(
                "camera=%s watchdog: no frame for %.1fs (limit %.1fs); "
                "releasing capture to unblock read()",
                self.camera_id,
                stale_for,
                self.cfg.read_timeout_s,
            )
            self._reconnect_requested.set()
            self._release()


class FileReader(_BaseReader):
    """Same interface over a local MP4.

    Integration tests run against fixture RTSP streams served by MediaMTX, but
    unit tests and the offline demo use this so they need neither a network nor
    a camera (CLAUDE.md/Testing).
    """

    def __init__(
        self,
        camera_id: str,
        path: str,
        cfg: IngestConfig | None = None,
        on_frame: OnFrame | None = None,
        on_state: OnState | None = None,
        *,
        loop: bool = True,
        realtime: bool = True,
    ) -> None:
        super().__init__(camera_id, cfg or IngestConfig(), on_frame or (lambda _f: None), on_state)
        self.path = path
        self.loop = loop
        self.realtime = realtime

    def _run(self) -> None:
        import cv2

        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.path)
            if not capture.isOpened():
                self.stats.last_error = f"cannot open {self.path}"
                self._set_state(StreamState.FAILED, self.stats.last_error)
                logger.error("camera=%s cannot open fixture %s", self.camera_id, self.path)
                return

            self._set_state(StreamState.LIVE)
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = 1.0 / source_fps

            while not self._stop.is_set():
                ok, image = capture.read()
                if not ok or image is None:
                    break
                self.stats.frames_read += 1
                now = time.monotonic()
                if self._should_emit(now):
                    h, w = image.shape[:2]
                    self._emit(image, w, h)
                else:
                    self.stats.frames_dropped_fps += 1
                    self._pending_gap += 1
                if self.realtime:
                    self._stop.wait(frame_interval)

            capture.release()
            if not self.loop:
                self._set_state(StreamState.FAILED, "fixture ended")
                return
            self.stats.reconnects += 1


def _redact(url: str) -> str:
    """Strip credentials before a URL reaches a log line (§12)."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***:***@{host}"


def build_reader(
    camera_id: str,
    source: str,
    cfg: IngestConfig,
    on_frame: OnFrame,
    on_state: OnState | None = None,
) -> _BaseReader:
    """Pick a reader by source. Anything that is not RTSP is treated as a file."""
    if source.startswith(("rtsp://", "rtsps://")):
        return RtspReader(camera_id, cfg, on_frame, on_state)
    return FileReader(camera_id, source, cfg, on_frame, on_state)
