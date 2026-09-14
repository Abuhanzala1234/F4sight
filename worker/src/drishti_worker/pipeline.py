"""Pipeline: thread topology, backpressure, and the per-camera stage
(BUILD_SPEC §7.14, §3.2, §3.3).

Thread model, and why it is threads (§3.2): the hot path is CPU/GPU-bound, and
every heavy call — ``cv2.VideoCapture.read``, ``ort.InferenceSession.run`` —
releases the GIL, so threads genuinely parallelise. ``asyncio`` would serialise
the same work behind more ceremony. One process rather than one-per-camera
because the accelerator is a single shared resource and N processes means N
model copies in VRAM.

    RtspReader ──▶ Queue[Frame] (maxsize=4, DROP-OLDEST)
                        │
                        ▼
              InferenceThread (shared across cameras, batched)
                        │
                        ▼
              Queue[DetBundle] (maxsize=16, BLOCKS)
                        │
                        ▼
       StageThread (per camera): track → geometry → rules → risk
                                 → debounce → evidence → sinks

The two queues have deliberately opposite policies:

* **The frame queue drops the oldest.** On a slow host we want to analyse
  *recent* frames, never a growing backlog of stale ones. An alert about where
  someone was thirty seconds ago is not an alert. Drops are counted and surfaced
  as reduced effective fps — being honest about degradation beats lying with
  latency.
* **The detection queue blocks.** A full detection queue means the stage thread
  is wedged, which is a bug we want to see rather than paper over.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .alerting import FrameBuffer
from .anpr import AnprConfig, PlateVoter
from .detect import Detector, DetectorConfig
from .enhance import EnhanceConfig, enhance_for_model
from .evqm import EVQM, EVQMConfig
from .ingest import IngestConfig, build_reader
from .risk import RiskConfig, RiskContext, score
from .rules import DebounceConfig, Debouncer, Decision, RuleConfig, RuleEngine
from .track import ByteTracker, TrackerConfig
from .types import (
    CameraRuntime,
    Detection,
    Frame,
    FrameTransform,
    StreamState,
    ZoneRuntime,
)
from .watchlist import WatchlistCache

logger = logging.getLogger(__name__)

__all__ = ["CameraWorker", "DetBundle", "Pipeline", "PipelineStats"]


@dataclass
class DetBundle:
    """One frame's detections, already mapped to ORIGINAL coordinates."""

    frame: Frame
    detections: list[Detection]
    transform: FrameTransform
    profile: str
    enhancement_params: Mapping[str, Any]
    inference_ms: float


@dataclass
class PipelineStats:
    frames_in: int = 0
    frames_dropped_queue: int = 0
    batches: int = 0
    inferences: int = 0
    alerts_emitted: int = 0
    alerts_suppressed: int = 0
    alerts_merged: int = 0
    last_inference_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_in": self.frames_in,
            "frames_dropped_queue": self.frames_dropped_queue,
            "batches": self.batches,
            "inferences": self.inferences,
            "alerts_emitted": self.alerts_emitted,
            "alerts_suppressed": self.alerts_suppressed,
            "alerts_merged": self.alerts_merged,
            "last_inference_ms": round(self.last_inference_ms, 2),
        }


class _DropOldestQueue(queue.Queue):
    """Bounded queue that evicts the oldest item instead of blocking.

    The whole point of §7.14's frame queue. ``dropped`` is read by the health
    endpoint so an operator can see the system is keeping up, or isn't.
    """

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize=maxsize)
        self.dropped = 0

    def put_latest(self, item: Any) -> None:
        while True:
            try:
                self.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass  # another thread drained it; retry the put


# ---------------------------------------------------------------------------
# Per-camera worker
# ---------------------------------------------------------------------------


class CameraWorker:
    """Owns one camera: reader, EVQM, tracker, rules, debouncer, stage thread."""

    def __init__(
        self,
        camera: CameraRuntime,
        source: str,
        zones: Sequence[ZoneRuntime],
        *,
        ingest_cfg: IngestConfig,
        evqm_cfg: EVQMConfig,
        enhance_cfg: EnhanceConfig,
        tracker_cfg: TrackerConfig,
        rule_cfg: RuleConfig,
        risk_cfg: RiskConfig,
        debounce_cfg: DebounceConfig,
        frame_queue: _DropOldestQueue,
        on_alert: Any,
        on_state: Any = None,
        on_tracks: Any = None,
        clip_pre_roll_s: float = 5.0,
        anpr_cfg: AnprConfig | None = None,
        anpr_reader: Any = None,
        watchlist: WatchlistCache | None = None,
        plate_hmac_key: bytes = b"",
    ) -> None:
        self.camera = camera
        self.zones = list(zones)
        self.evqm = EVQM(evqm_cfg, camera.camera_id)
        self.enhance_cfg = enhance_cfg
        self.tracker = ByteTracker(tracker_cfg, camera.camera_id)
        self.rules = RuleEngine(rule_cfg, risk_cfg)
        self.debouncer = Debouncer(debounce_cfg)
        self.risk_cfg = risk_cfg
        self._frame_queue = frame_queue
        self._det_queue: queue.Queue[DetBundle] = queue.Queue(maxsize=16)
        self._on_alert = on_alert
        self._on_tracks = on_tracks
        self._stop = threading.Event()
        self._stage_thread: threading.Thread | None = None
        self.stats = PipelineStats()

        # ORIGINAL frames only (P4), pushed in ``_on_frame`` before enhancement.
        # Sized with headroom over the pre-roll window so a momentary fps dip
        # does not truncate the clip an alert is about to ask for.
        margin = 1.5
        buffer_frames = max(2, int(clip_pre_roll_s * camera.analytics_fps * margin))
        self.frame_buffer = FrameBuffer(max_frames=buffer_frames)

        # ANPR (§7.9). ``anpr_reader`` is shared across every camera (built
        # once in __main__.py, same reasoning as the detector: one accelerator,
        # not one model copy per camera). ``_plate_voters`` is per-camera,
        # per-track state living outside the tracker — same pattern as
        # RuleEngine's ``_states`` — and is cleared in ``_process`` whenever
        # the tracker reports a track closed, so a plate never outlives the
        # vehicle it was read from.
        self.anpr_cfg = anpr_cfg
        self._anpr_reader = anpr_reader if (anpr_cfg and anpr_cfg.enabled) else None
        self._plate_voters: dict[int, PlateVoter] = {}
        self._plate_hits: dict[int, Mapping[str, Any]] = {}
        self._watchlist = watchlist
        self._plate_hmac_key = plate_hmac_key

        self.reader = build_reader(camera.camera_id, source, ingest_cfg, self._on_frame, on_state)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._stage_thread = threading.Thread(
            target=self._stage_loop, name=f"stage-{self.camera.code}", daemon=True
        )
        self._stage_thread.start()
        self.reader.start()
        logger.info(
            "camera=%s worker started (%d zones, %.1f fps analytics)",
            self.camera.code,
            len(self.zones),
            self.camera.analytics_fps,
        )

    def stop(self, timeout_s: float = 5.0) -> None:
        self.reader.stop(timeout_s)
        self._stop.set()
        if self._stage_thread is not None:
            self._stage_thread.join(timeout=timeout_s)

    @property
    def state(self) -> StreamState:
        return self.reader.state

    def submit_detections(self, bundle: DetBundle) -> None:
        """Called by the shared inference thread. BLOCKS when full, by design."""
        self._det_queue.put(bundle, timeout=5.0)

    # -- frame intake ------------------------------------------------------

    def _on_frame(self, frame: Frame) -> None:
        """Reader callback. Samples EVQM, then queues the frame for inference.

        EVQM runs here, on the reader thread, because it is cheap (< 3 ms on a
        320-px downscale) and its answer selects the enhancement profile.

        Enhancement does NOT run here. It happens in the inference thread AFTER
        the letterbox resize — see Pipeline._run_batch. Enhancing the full frame
        first means processing 921k pixels and discarding three quarters of them
        in the resize, which measured 50 ms per frame against an 8 ms budget.

        The frame buffer is pushed here too, deliberately before enhancement:
        it is what a clip is cut from, and evidence is the ORIGINAL frame (P4).
        """
        self.stats.frames_in += 1
        self.frame_buffer.push(frame)
        self.evqm.observe(frame)
        self._frame_queue.put_latest((self, frame, self.evqm.profile, self.enhance_cfg))

    # -- stage thread ------------------------------------------------------

    def _stage_loop(self) -> None:
        while not self._stop.is_set():
            try:
                bundle = self._det_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(bundle)
            except Exception:
                # One bad frame must never kill a camera (P8).
                logger.exception(
                    "camera=%s stage failed on frame_id=%s; continuing",
                    self.camera.code,
                    bundle.frame.frame_id,
                )

    def _process(self, bundle: DetBundle) -> None:
        frame = bundle.frame
        tracks = self.tracker.update(bundle.detections, frame.ts_utc)

        # Track teardown: this is where per-track state is destroyed, including
        # face embeddings that matched nothing (P6, §7.5).
        for closed in self.tracker.close_expired():
            self.rules.close_track(closed.track_id)
            self.debouncer.close_track(self.camera.camera_id, closed.track_id)
            self._plate_voters.pop(closed.track_id, None)
            self._plate_hits.pop(closed.track_id, None)

        plate_hit = self._read_plates(tracks, frame)

        # Live overlay feed: every processed frame, not just ones that raise a
        # signal, and BEFORE the no-signals early-return below -- an operator
        # watching the wall should see every real track the model is actually
        # following, not just the ones that happened to cross a rule.
        if self._on_tracks is not None:
            confirmed = [t for t in tracks if t.is_confirmed]
            if confirmed:
                self._on_tracks(
                    self.camera.camera_id,
                    frame.ts_utc,
                    [
                        {
                            "track_id": t.track_id,
                            "cls": t.cls,
                            "box": [round(v, 1) for v in t.box],
                            "speed_px_s": round(t.speed_px_s(self.camera.analytics_fps), 1),
                        }
                        for t in confirmed
                    ],
                )

        signals_by_track = self.rules.evaluate(
            tracks,
            self.camera,
            self.zones,
            frame.ts_utc,
            bundle.profile,
            tamper_suspected=self.evqm.tamper_suspected,
            plate_hit=plate_hit,
        )
        if not signals_by_track:
            return

        by_id = {t.track_id: t for t in tracks}
        for track_id, signals in signals_by_track.items():
            track = by_id.get(track_id)
            if track is None:
                continue

            # The alert's kind is the STRONGEST signal, not whichever rule
            # happens to be registered first. An intrusion that also triggered
            # a weaker approach signal is an intrusion.
            primary = max(signals, key=lambda s: s.weight)
            alert_id = str(uuid.uuid4())
            decision = self.debouncer.submit(
                camera_id=self.camera.camera_id,
                track_id=track_id,
                zone_id=primary.detail.get("zone_id"),
                rule_code=primary.code,
                now=frame.ts_utc,
                alert_id=alert_id,
                weight=primary.weight,
            )

            if decision.decision is Decision.SUPPRESS:
                self.stats.alerts_suppressed += 1
                continue
            if decision.decision is Decision.MERGE:
                self.stats.alerts_merged += 1
                continue
            if decision.decision is Decision.RATE_LIMITED:
                self.stats.alerts_suppressed += 1
                continue

            risk = score(
                signals,
                RiskContext(
                    config=self.risk_cfg,
                    evqm_profile=bundle.profile,
                    track_max_conf=track.max_conf,
                    track_age_frames=track.age_frames,
                ),
            )
            self.stats.alerts_emitted += 1
            self._on_alert(
                alert_id=alert_id,
                camera=self.camera,
                track=track,
                frame=frame,
                signals=signals,
                risk=risk,
                profile=bundle.profile,
                enhancement_params=bundle.enhancement_params,
                escalated=decision.decision is Decision.ESCALATE,
                suppressed_since_last=decision.suppressed_count,
                frame_buffer=self.frame_buffer,
            )

    def _read_plates(self, tracks: Sequence[Any], frame: Frame) -> Mapping[str, Any] | None:
        """ANPR (§7.9): read + vote for every vehicle track, off the hot path
        in the sense that matters -- it costs nothing when disabled or when no
        vehicle track is present, and a single OCR pass is cheap next to
        detection.

        Returns at most one settled watchlist hit. ``RuleContext.plate_hit``
        is a single slot, not a per-track map (§7.7, a frozen contract) --
        two vehicles both settling a watchlist plate in the same frame is
        rare enough that surfacing the first and picking up the second on its
        next settled frame is the honest trade-off, not a silent drop.

        A resolved hit is cached in ``_plate_hits`` and returned on every
        subsequent call, not just the frame it settled on. It has to be: OCR
        voting can settle a plate (``min_frames_agreed``, typically 3) well
        before the rule gate opens (``min_track_age_frames``, default 8) --
        without the cache, the one frame where ``plate_hit`` was non-None is
        never one the gate lets through, and WATCHLIST_PLATE can never fire.
        """
        if self._anpr_reader is None:
            return None
        cfg = self.anpr_cfg
        classes = cfg.classes if cfg else ("vehicle",)
        height, width = frame.image.shape[:2]

        for track in tracks:
            if track.cls not in classes:
                continue

            known = self._plate_hits.get(track.track_id)
            if known is not None:
                return known

            voter = self._plate_voters.get(track.track_id)
            if voter is None:
                voter = PlateVoter(
                    min_frames_agreed=cfg.min_frames_agreed if cfg else 3,
                    min_char_conf=cfg.min_char_conf if cfg else 0.55,
                    window_frames=cfg.window_frames if cfg else 30,
                    max_candidates=cfg.max_crops_per_track if cfg else 12,
                )
                self._plate_voters[track.track_id] = voter
            if voter.settled is not None:
                continue  # already read this vehicle; no need to keep cropping it

            x1, y1, x2, y2 = (int(max(0, v)) for v in track.box)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            try:
                candidate = self._anpr_reader.read(frame.image[y1:y2, x1:x2], frame.frame_id)
            except Exception:
                # A bad crop or a flaky OCR backend must not cost the frame,
                # let alone the camera (P8).
                logger.exception(
                    "camera=%s ANPR read failed for track=%s; skipping this crop",
                    self.camera.code,
                    track.track_id,
                )
                continue
            if candidate is None:
                continue

            settled = voter.add(candidate)
            if settled is None or self._watchlist is None or not self._plate_hmac_key:
                continue
            try:
                digest = settled.hmac(self._plate_hmac_key)
            except ValueError:
                logger.exception(
                    "camera=%s could not HMAC the settled plate for track=%s",
                    self.camera.code,
                    track.track_id,
                )
                continue
            hit = self._watchlist.plate_match(digest)
            if hit is not None:
                resolved = {
                    "track_id": track.track_id,
                    "plate_hmac": digest,
                    "category": hit.category,
                    "frames_agreed": settled.frames_agreed,
                }
                self._plate_hits[track.track_id] = resolved
                return resolved
        return None

    def health(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera.camera_id,
            "camera_code": self.camera.code,
            "state": str(self.reader.state),
            "evqm_profile": self.evqm.profile,
            "evqm_metrics": self.evqm.metrics.as_dict() if self.evqm.metrics else None,
            "tracks_active": self.tracker.active_count,
            "ingest": self.reader.stats.as_dict(),
            "pipeline": self.stats.as_dict(),
            "debounce": self.debouncer.stats(),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Owns the shared inference thread and every camera worker."""

    def __init__(
        self,
        detector: Detector,
        detector_cfg: DetectorConfig,
        *,
        frame_queue_size: int = 4,
        stats_interval_s: float = 10.0,
    ) -> None:
        self.detector = detector
        self.detector_cfg = detector_cfg
        self.frame_queue = _DropOldestQueue(frame_queue_size)
        self.workers: dict[str, CameraWorker] = {}
        self.stats = PipelineStats()
        self._stop = threading.Event()
        self._infer_thread: threading.Thread | None = None
        self._stats_thread: threading.Thread | None = None
        self._stats_interval = stats_interval_s
        self._started_at = time.monotonic()

    def add_worker(self, worker: CameraWorker) -> None:
        self.workers[worker.camera.camera_id] = worker

    def remove_worker(self, camera_id: str, timeout_s: float = 5.0) -> None:
        """Stop and drop one camera while the pipeline keeps running.

        The counterpart to hot-adding (``add_worker`` + ``worker.start()``):
        this is what lets a camera be disconnected from the UI without
        touching any other camera or the shared inference thread.
        """
        worker = self.workers.pop(camera_id, None)
        if worker is not None:
            worker.stop(timeout_s)

    def start(self) -> None:
        self._stop.clear()
        self._infer_thread = threading.Thread(
            target=self._infer_loop, name="inference", daemon=True
        )
        self._infer_thread.start()
        self._stats_thread = threading.Thread(target=self._stats_loop, name="stats", daemon=True)
        self._stats_thread.start()
        for worker in self.workers.values():
            worker.start()
        logger.info(
            "pipeline started: %d cameras, detector=%s, batch<=%d",
            len(self.workers),
            self.detector.backend,
            self.detector_cfg.max_batch,
        )

    def stop(self, timeout_s: float = 5.0) -> None:
        for worker in self.workers.values():
            worker.stop(timeout_s)
        self._stop.set()
        for thread in (self._infer_thread, self._stats_thread):
            if thread is not None:
                thread.join(timeout=timeout_s)
        logger.info("pipeline stopped")

    # -- inference ---------------------------------------------------------

    def _infer_loop(self) -> None:
        """Drain up to ``max_batch`` frames, or fire after ``max_wait_ms``.

        Never wait for a full batch. On a two-camera laptop a full batch of 8
        would mean holding the first frame for over a second, and latency is
        what an operator actually experiences.
        """
        max_batch = max(1, self.detector_cfg.max_batch)
        max_wait = self.detector_cfg.max_wait_ms / 1000.0

        while not self._stop.is_set():
            batch: list[tuple[CameraWorker, Frame, str, Any]] = []
            deadline = time.monotonic() + max_wait
            try:
                batch.append(self.frame_queue.get(timeout=0.5))
            except queue.Empty:
                continue

            while len(batch) < max_batch and time.monotonic() < deadline:
                try:
                    batch.append(self.frame_queue.get_nowait())
                except queue.Empty:
                    time.sleep(0.001)

            self._run_batch(batch)

    def _run_batch(self, batch: Sequence[tuple[CameraWorker, Frame, str, Any]]) -> None:
        """Letterbox, enhance, infer.

        Order matters, and it is: resize FIRST, then enhance, then pad.

        * Resizing first means enhancement touches ~230k pixels instead of 921k.
          Measured: full-frame fog dehaze 50 ms, post-resize 6 ms (§7.3 budget
          is 8 ms). Enhancement is *for the model*, so it belongs at model scale.
        * Enhancing before padding matters too: CLAHE equalises a histogram, and
          the black letterbox bars would skew it badly.
        * The ORIGINAL frame is untouched throughout and is what evidence uses (P4).
        """
        import cv2
        import numpy as np

        model_w, model_h = self.detector.input_size
        images: list[Any] = []
        transforms: list[FrameTransform] = []
        enhancements: list[Any] = []

        for _worker, frame, profile, enhance_cfg in batch:
            transform = FrameTransform.letterbox((frame.width, frame.height), (model_w, model_h))
            resized = cv2.resize(
                frame.image,
                (
                    int(frame.width * transform.scale_x),
                    int(frame.height * transform.scale_y),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
            enhanced = enhance_for_model(resized, profile, enhance_cfg)
            canvas = np.zeros((model_h, model_w, 3), dtype=np.uint8)
            y0, x0 = int(transform.pad_y), int(transform.pad_x)
            canvas[y0 : y0 + enhanced.image.shape[0], x0 : x0 + enhanced.image.shape[1]] = (
                enhanced.image
            )
            images.append(canvas)
            transforms.append(transform)
            enhancements.append(enhanced)

        started = time.monotonic()
        try:
            raw_batches = self.detector.infer(images)
        except Exception:
            logger.exception("inference failed for a batch of %d frames; dropping it", len(batch))
            return
        elapsed_ms = (time.monotonic() - started) * 1000.0

        self.stats.batches += 1
        self.stats.inferences += len(batch)
        self.stats.last_inference_ms = elapsed_ms

        for (worker, frame, profile, _cfg), transform, enhanced, raws in zip(
            batch, transforms, enhancements, raw_batches, strict=True
        ):
            # THE INVARIANT: map to original-frame coordinates here, once, before
            # anything downstream can see a model-space box.
            detections = self._to_detections(raws, transform)
            try:
                worker.submit_detections(
                    DetBundle(
                        frame=frame,
                        detections=detections,
                        transform=transform,
                        profile=profile,
                        enhancement_params=enhanced.params,
                        inference_ms=elapsed_ms / len(batch),
                    )
                )
            except queue.Full:
                logger.error(
                    "camera=%s detection queue is full; the stage thread is wedged",
                    worker.camera.code,
                )

    def _to_detections(self, raws: Sequence[Any], transform: FrameTransform) -> list[Detection]:
        cfg = self.detector_cfg
        out: list[Detection] = []
        for raw in raws:
            mapped = cfg.class_map.get(raw.cls_id)
            if mapped is None:
                continue  # not in our taxonomy; dropped before the tracker (§7.4)
            cls = str(mapped.get("cls", "unknown"))
            if raw.conf < cfg.threshold_for(cls):
                continue
            box = transform.to_original(raw.box)
            width, height = box[2] - box[0], box[3] - box[1]
            if height < cfg.min_box_height_px or width * height < cfg.min_box_area_px:
                continue  # at range this is noise, not an object
            out.append(
                Detection(
                    cls=cls,
                    conf=raw.conf,
                    box=box,
                    cls_id=raw.cls_id,
                    attributes={k: v for k, v in mapped.items() if k != "cls"},
                )
            )
        return out

    # -- observability -----------------------------------------------------

    def _stats_loop(self) -> None:
        while not self._stop.wait(self._stats_interval):
            for worker in self.workers.values():
                health = worker.health()
                logger.info(
                    "camera=%s state=%s profile=%s fps_in=%.1f tracks=%d "
                    "alerts=%d suppressed=%d infer=%.0fms dropped=%d",
                    health["camera_code"],
                    health["state"],
                    health["evqm_profile"],
                    health["ingest"]["fps_in"],
                    health["tracks_active"],
                    health["pipeline"]["alerts_emitted"],
                    health["pipeline"]["alerts_suppressed"],
                    self.stats.last_inference_ms,
                    self.frame_queue.dropped,
                )
            if self.frame_queue.dropped > 0:
                logger.warning(
                    "frame queue has dropped %d frames since start; "
                    "effective analytics fps is below the configured rate",
                    self.frame_queue.dropped,
                )

    def health(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.monotonic() - self._started_at, 1),
            "detector": self.detector.backend,
            "frames_dropped_queue": self.frame_queue.dropped,
            "pipeline": self.stats.as_dict(),
            "cameras": [w.health() for w in self.workers.values()],
        }
