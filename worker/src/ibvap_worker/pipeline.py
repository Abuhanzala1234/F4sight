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
from dataclasses import dataclass, replace
from typing import Any

from .activity import ActivityConfig, ActivityGate
from .alerting import FrameBuffer
from .anpr import AnprConfig, PlateVoter
from .detect import Detector, DetectorConfig
from .enhance import EnhanceConfig, enhance_for_model
from .evqm import EVQM, EVQMConfig
from .faces import FaceConfig, FaceVoter
from .gesture import GestureConfig, GestureVoter, Keypoint, Pose, classify_gesture
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
from .watchlist import FaceWatchlistCache, WatchlistCache
from .weapon import WeaponConfig, WeaponVoter

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


def _activity_config_for(block: Mapping[str, Any], *, window_frames: int) -> ActivityConfig:
    """Build a gate config, defaulting its warm-up to the model's voting window.

    The default is not a tuning choice, it is a correctness one: the gate must
    let the model look at a track at least a full voting window before it is
    allowed to start skipping, or the vote never settles and a motionless
    subject is never confirmed (activity.py, guard 3). An explicit value in
    config still wins -- but it should be >= the window, and lowering it is how
    you reintroduce the bug.
    """
    cfg = ActivityConfig.from_mapping({"activity_gate": block})
    if "min_runs_before_skip" in dict(block or {}):
        return cfg
    return replace(cfg, min_runs_before_skip=max(1, window_frames))


def _zone_signature(zones: Sequence[ZoneRuntime]) -> tuple[Any, ...]:
    """Everything about a zone set that changes what it alerts on.

    Compared by the reload loop to decide whether a running camera's zones
    actually changed. Deliberately excludes ``name``: renaming a zone is not a
    reason to swap the geometry the stage thread is reading.
    """
    return tuple(
        (
            z.zone_id,
            z.kind,
            z.polygon,
            z.direction,
            tuple(z.classes),
            z.severity_base,
            z.enabled,
        )
        for z in zones
    )


def _crop_letterboxed(
    frame: Frame, box: tuple[float, float, float, float], input_size: tuple[int, int]
) -> tuple[Any, FrameTransform, tuple[int, int]] | None:
    """Crop a track's box out of a frame and letterbox it for a second-stage model.

    Shared by every crop-fed stage (gestures, weapons). Returns the padded
    canvas, the transform that produced it, and the crop's origin in the
    original frame — the caller needs all three to map the model's answer back,
    since ``FrameTransform``'s ``crop_x``/``crop_y`` exist for exactly this
    "tile within the frame" case. Returns None for a degenerate box rather than
    handing a zero-width array to a model.

    The two ``int()`` truncations match the ones FrameTransform.letterbox uses
    to derive its padding, for the same reason as in ``Pipeline._run_batch``:
    disagree and the answer comes back off by a pixel.
    """
    import cv2
    import numpy as np

    model_w, model_h = input_size
    height, width = frame.image.shape[:2]
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    transform = FrameTransform.letterbox((x2 - x1, y2 - y1), (model_w, model_h))
    resized = cv2.resize(
        frame.image[y1:y2, x1:x2],
        (int((x2 - x1) * transform.scale_x), int((y2 - y1) * transform.scale_y)),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros((model_h, model_w, 3), dtype=np.uint8)
    oy, ox = int(transform.pad_y), int(transform.pad_x)
    canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    return canvas, transform, (x1, y1)


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
        gesture_cfg: GestureConfig | None = None,
        pose_estimator: Any = None,
        weapon_cfg: WeaponConfig | None = None,
        weapon_detector: Any = None,
        face_cfg: FaceConfig | None = None,
        face_detector: Any = None,
        face_embedder: Any = None,
        face_watchlist: FaceWatchlistCache | None = None,
        watchlist: WatchlistCache | None = None,
        plate_hmac_key: bytes = b"",
    ) -> None:
        self.camera = camera
        self.zones = list(zones)
        # Baselined from the zones we were built with, so the reload loop's
        # first poll compares against what is actually loaded and does not
        # report a change that never happened.
        self._zone_signature = _zone_signature(zones)
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

        # Hand signals (§7.7). Same shape as ANPR above: one shared estimator,
        # per-track voter state, and None when disabled so the whole feature
        # costs exactly nothing.
        self.gesture_cfg = gesture_cfg
        self._pose = pose_estimator if (gesture_cfg and gesture_cfg.enabled) else None
        self._gesture_voters: dict[int, GestureVoter] = {}
        # Cheap activity gate + the last settled answer it lets us reuse. The
        # cache is what stops stillness from clearing a confirmed state; see
        # activity.py's module docstring.
        self._gesture_gate = ActivityGate(
            _activity_config_for(
                gesture_cfg.activity_gate if gesture_cfg else {},
                window_frames=gesture_cfg.window_frames if gesture_cfg else 8,
            )
        )
        self._gesture_last: dict[int, dict[str, Any]] = {}

        # Weapons (§7.7). Third instance of the same second-stage shape.
        self.weapon_cfg = weapon_cfg
        self._weapon = weapon_detector if (weapon_cfg and weapon_cfg.enabled) else None
        self._weapon_voters: dict[int, WeaponVoter] = {}
        self._weapon_gate = ActivityGate(
            _activity_config_for(
                weapon_cfg.activity_gate if weapon_cfg else {},
                window_frames=weapon_cfg.window_frames if weapon_cfg else 8,
            )
        )
        self._weapon_last: dict[int, dict[str, Any]] = {}

        # Faces (§7.10). OPT-IN -- see faces.py's module docstring. Fourth
        # instance of the same second-stage shape, including the held-state
        # cache: a matched person who stands still is still that person, and
        # without the cache the gate's own skip would make them stop being
        # reported the moment they stop moving -- the same starvation the
        # weapon path's tests caught, just with identity instead of ARMED.
        self.face_cfg = face_cfg
        self._face_detector = face_detector if (face_cfg and face_cfg.enabled) else None
        self._face_embedder = face_embedder if (face_cfg and face_cfg.enabled) else None
        self._face_watchlist = face_watchlist
        self._face_voters: dict[int, FaceVoter] = {}
        self._face_gate = ActivityGate(
            _activity_config_for(
                face_cfg.activity_gate if face_cfg else {},
                window_frames=face_cfg.window_frames if face_cfg else 6,
            )
        )
        self._face_last: dict[int, dict[str, Any]] = {}

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
        self._reconcile_frame_size(frame)
        self.frame_buffer.push(frame)
        self.evqm.observe(frame)
        self._frame_queue.put_latest((self, frame, self.evqm.profile, self.enhance_cfg))

    def _reconcile_frame_size(self, frame: Frame) -> None:
        """Re-project zone geometry onto the size the camera ACTUALLY sends.

        Zones are stored normalised (§6.2) and were denormalised at startup
        against the camera row's ``resolution_w/h``. Nothing updates that
        column when an operator binds a real camera by IP, so it is still the
        seeded 1280x720 for every live camera. A phone streaming 640x480 got
        every zone denormalised to twice its own frame -- tripwires and areas
        landing off the picture entirely, costing intrusion alerts with no
        error anywhere. The decoded frame is the only real authority on this.

        Rescaling the already-denormalised pixels by the ratio is the same
        arithmetic as denormalising against the true size, and it keeps the
        normalised originals out of the hot path.

        Runs on the reader thread while ``_process`` reads these on the stage
        thread, so both attributes are REPLACED, never mutated in place: a
        stage-thread frame sees either the old geometry or the new one, never
        a half-rescaled polygon.
        """
        if frame.width == self.camera.width and frame.height == self.camera.height:
            return
        sx = frame.width / self.camera.width
        sy = frame.height / self.camera.height
        logger.warning(
            "camera=%s streams %dx%d, not the configured %dx%d; rescaling %d zone(s)",
            self.camera.code,
            frame.width,
            frame.height,
            self.camera.width,
            self.camera.height,
            len(self.zones),
        )
        self.zones = [
            replace(z, polygon=tuple((x * sx, y * sy) for x, y in z.polygon)) for z in self.zones
        ]
        self.camera = replace(self.camera, width=frame.width, height=frame.height)

    def update_zones_if_changed(
        self, zones: Sequence[ZoneRuntime], nominal: tuple[int, int]
    ) -> bool:
        """Swap in a new zone set on a RUNNING camera. Returns True if it changed.

        A zone is the policy decision about what counts as an intrusion, so an
        admin editing one expects the running system to start honouring it --
        the same "no restart required" contract the camera hot-add/hot-drop
        path already keeps. Without this, a zone drawn today was only picked up
        the next time the camera (or the worker) happened to restart.

        Zones arrive denormalised against the camera row's NOMINAL resolution,
        but this worker may already have discovered the camera actually streams
        something else and rescaled its own copy (``_reconcile_frame_size``).
        Re-apply that same ratio here, or a freshly drawn zone lands in the
        wrong place on exactly the cameras that needed the correction most.

        Assigns a new list rather than mutating the existing one, for the same
        reason ``_reconcile_frame_size`` does: the stage thread reads
        ``self.zones`` concurrently and must always see one coherent set.
        """
        signature = _zone_signature(zones)
        if signature == self._zone_signature:
            return False

        nom_w, nom_h = nominal
        scaled = list(zones)
        if nom_w and nom_h and (nom_w != self.camera.width or nom_h != self.camera.height):
            sx = self.camera.width / nom_w
            sy = self.camera.height / nom_h
            scaled = [
                replace(z, polygon=tuple((x * sx, y * sy) for x, y in z.polygon)) for z in zones
            ]

        self.zones = scaled
        self._zone_signature = signature
        return True

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
            self._gesture_voters.pop(closed.track_id, None)
            self._weapon_voters.pop(closed.track_id, None)
            self._gesture_gate.close_track(closed.track_id)
            self._weapon_gate.close_track(closed.track_id)
            self._gesture_last.pop(closed.track_id, None)
            self._weapon_last.pop(closed.track_id, None)
            self._face_voters.pop(closed.track_id, None)
            self._face_gate.close_track(closed.track_id)
            self._face_last.pop(closed.track_id, None)

        plate_hit = self._read_plates(tracks, frame)
        gesture_hit = self._read_gestures(tracks, frame)
        weapon_hit = self._read_weapons(tracks, frame)
        face_hit = self._read_faces(tracks, frame)

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
                    # The boxes above are in THIS frame's pixel space, so the
                    # dashboard needs this frame's size to un-project them --
                    # it cannot use the camera row's resolution, which is a
                    # static guess written at seed time and wrong for any
                    # camera that negotiated something else (a phone in
                    # portrait is the common case). Travels per-frame rather
                    # than being stored once, so a camera that changes
                    # resolution mid-run corrects itself on the next frame.
                    frame.width,
                    frame.height,
                )

        signals_by_track = self.rules.evaluate(
            tracks,
            self.camera,
            self.zones,
            frame.ts_utc,
            bundle.profile,
            tamper_suspected=self.evqm.tamper_suspected,
            plate_hit=plate_hit,
            gesture_hit=gesture_hit,
            weapon_hit=weapon_hit,
            face_hit=face_hit,
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

    def _read_weapons(self, tracks: Sequence[Any], frame: Frame) -> Mapping[str, Any] | None:
        """Weapons (§7.7): a two-class detector on person-track crops, voted.

        Structurally identical to ``_read_gestures`` -- same crop, same cadence
        control, same single-slot return -- with one deliberate difference:
        ties are impossible here because the vote is on ARMED, not on which
        weapon (see weapon.py). Never latches, for the same reason a gesture
        does not: somebody can put something down.
        """
        if self._weapon is None:
            return None
        cfg = self.weapon_cfg
        if cfg is None:
            return None
        if frame.frame_id % cfg.every_n_frames:
            return None

        best: tuple[float, dict[str, Any]] | None = None
        candidates = [t for t in tracks if t.cls in cfg.classes and t.is_confirmed]
        for track in candidates[: max(0, cfg.max_tracks_per_frame)]:
            prepared = _crop_letterboxed(frame, track.box, cfg.input_size)
            if prepared is None:
                continue
            canvas, _transform, _origin = prepared

            # Cheap gate in front of the expensive model (activity.py). A crop
            # that has not changed since we last looked does not need another
            # 33 ms of inference to tell us so.
            run, _reason = self._weapon_gate.check(track.track_id, canvas, frame.frame_id)
            if not run:
                # THE safety rule: stillness must never disarm somebody. The
                # gate decides whether to spend an inference, never that a
                # previous answer expired, so a confirmed weapon keeps being
                # reported while its owner stands motionless.
                held = self._weapon_last.get(track.track_id)
                if held is not None and (best is None or held["conf"] > best[0]):
                    best = (held["conf"], held)
                continue

            try:
                candidate = self._weapon.detect(canvas)
            except Exception:
                # P8: a flaky accelerator must not cost the frame or the camera.
                logger.exception(
                    "camera=%s weapon detection failed for track=%s; skipping this crop",
                    self.camera.code,
                    track.track_id,
                )
                continue

            voter = self._weapon_voters.get(track.track_id)
            if voter is None:
                voter = WeaponVoter(
                    min_frames_agreed=cfg.min_frames_agreed,
                    window_frames=cfg.window_frames,
                )
                self._weapon_voters[track.track_id] = voter

            settled = voter.add(candidate)
            if settled is None:
                # A real re-check that came back negative clears the held
                # state -- this is the path by which somebody who puts a
                # weapon down stops being armed.
                self._weapon_last.pop(track.track_id, None)
                continue
            record = {
                "track_id": track.track_id,
                "weapon_type": settled.cls,
                "conf": round(float(settled.conf), 3),
                "frames_agreed": int(settled.frames_agreed),
                "frames_seen": int(settled.frames_seen),
            }
            self._weapon_last[track.track_id] = record
            if best is None or settled.conf > best[0]:
                best = (settled.conf, record)

        return best[1] if best else None

    def _read_faces(self, tracks: Sequence[Any], frame: Frame) -> Mapping[str, Any] | None:
        """Faces (§7.10): detect + align + embed + match, on person crops.

        OPT-IN and fails closed on every missing piece: no detector, no
        embedder, or no watchlist loaded and this returns None immediately --
        there is no path by which a partially-configured face stage produces
        a match. Structurally the same crop/gate/vote shape as gestures and
        weapons, with the held-state cache (see the constructor's comment for
        why it belongs here too) and one addition unique to this stage: the
        match itself, not just a classification, so there is an extra step
        between "the model ran" and "we have a candidate" -- align the best
        face the detector found, embed it, then check it against the
        watchlist. A crop with no face, or a face too small to trust
        (``min_face_px``), produces a candidate of None, same as gestures
        report "no gesture this frame".
        """
        detector, embedder, watchlist = (
            self._face_detector,
            self._face_embedder,
            self._face_watchlist,
        )
        if detector is None or embedder is None:
            return None
        if watchlist is None or watchlist.count == 0:
            return None
        cfg = self.face_cfg
        if cfg is None:
            return None
        if frame.frame_id % cfg.every_n_frames:
            return None

        best: tuple[float, dict[str, Any]] | None = None
        candidates = [t for t in tracks if t.cls in cfg.classes and t.is_confirmed]
        for track in candidates[: max(0, cfg.max_tracks_per_frame)]:
            prepared = _crop_letterboxed(frame, track.box, cfg.detector_input_size)
            if prepared is None:
                continue
            canvas, _crop_transform, _crop_origin = prepared

            run, _reason = self._face_gate.check(track.track_id, canvas, frame.frame_id)
            if not run:
                held = self._face_last.get(track.track_id)
                if held is not None and (best is None or held["similarity"] > best[0]):
                    best = (held["similarity"], held)
                continue

            try:
                match = self._detect_and_match_face(canvas, cfg, detector, embedder, watchlist)
            except Exception:
                # P8: a flaky accelerator must not cost the frame or the camera.
                logger.exception(
                    "camera=%s face matching failed for track=%s; skipping this crop",
                    self.camera.code,
                    track.track_id,
                )
                continue

            voter = self._face_voters.get(track.track_id)
            if voter is None:
                voter = FaceVoter(
                    min_frames_agreed=cfg.min_frames_agreed,
                    window_frames=cfg.window_frames,
                )
                self._face_voters[track.track_id] = voter

            settled = voter.add(match)
            if settled is None:
                # A face that stops matching (turned away, walked out of
                # frame) must stop being reported -- same rule as a weapon
                # put down.
                self._face_last.pop(track.track_id, None)
                continue
            record = {
                "track_id": track.track_id,
                "ref_code": settled.ref_code,
                "category": settled.category,
                "similarity": round(float(settled.similarity), 4),
                "frames_agreed": int(settled.frames_agreed),
            }
            self._face_last[track.track_id] = record
            if best is None or settled.similarity > best[0]:
                best = (settled.similarity, record)

        return best[1] if best else None

    def _detect_and_match_face(
        self, canvas: Any, cfg: FaceConfig, detector: Any, embedder: Any, watchlist: Any
    ) -> Any:
        """One crop, start to finish: best face -> aligned -> embedded -> matched.

        Takes the detector/embedder/watchlist as arguments rather than reading
        ``self._face_*`` again -- the caller already narrowed them out of
        ``| None``, and re-reading the attributes here would just hand mypy
        (and a reader) the same union back.

        Everything here stays in the CROP's own coordinate space -- unlike
        gestures, nothing downstream needs a face's position in the original
        frame, only its identity, so there is no FrameTransform round trip to
        get right.
        """
        faces = detector.detect(canvas)
        if not faces:
            return None
        usable = [f for f in faces if min(f.width, f.height) >= cfg.min_face_px]
        if not usable:
            return None
        best_face = max(usable, key=lambda f: f.score)

        from .detect.onnx_face import align_face

        aligned = align_face(canvas, best_face.landmarks)
        embedding = embedder.embed(aligned)
        return watchlist.match(embedding)

    def _read_gestures(self, tracks: Sequence[Any], frame: Frame) -> Mapping[str, Any] | None:
        """Hand signals (§7.7): pose on person-track crops, voted over frames.

        Returns at most one settled gesture, matching ``plate_hit``'s single
        slot in the frozen RuleContext contract (§7.7). Two people signalling
        in the same frame is rare enough that surfacing the strongest and
        picking the other up on its next settled frame is an honest trade,
        not a silent drop.

        Unlike a plate, a gesture never latches: a person lowers their hands,
        and the voter's rolling window says so on its own. That is why there is
        no ``_gesture_hits`` cache to mirror ``_plate_hits`` -- caching a
        posture would leave HANDS_UP attached to somebody for the rest of their
        track.
        """
        if self._pose is None:
            return None
        cfg = self.gesture_cfg
        if cfg is None:
            return None
        # A posture is held over a second or two, so there is nothing to gain
        # from running the model on every frame -- and on a 4-camera wall this
        # would otherwise be the most expensive thing the worker does.
        if frame.frame_id % cfg.every_n_frames:
            return None

        best: tuple[float, dict[str, Any]] | None = None
        candidates = [t for t in tracks if t.cls in cfg.classes and t.is_confirmed]
        for track in candidates[: max(0, cfg.max_tracks_per_frame)]:
            prepared = _crop_letterboxed(frame, track.box, cfg.input_size)
            if prepared is None:
                continue
            canvas, transform, (x1, y1) = prepared

            # Same cheap gate as the weapon path (activity.py). A held gesture
            # is a still crop, so the held-state rule matters here too: hands
            # kept up must not stop reading as HANDS_UP once they stop moving.
            run, _reason = self._gesture_gate.check(track.track_id, canvas, frame.frame_id)
            if not run:
                held = self._gesture_last.get(track.track_id)
                if held is not None and (best is None or held["conf"] > best[0]):
                    best = (held["conf"], held)
                continue

            try:
                pose = self._pose.estimate(canvas)
            except Exception:
                # A flaky accelerator must not cost the frame, let alone the
                # camera (P8) -- the same contract the ANPR crop path keeps.
                logger.exception(
                    "camera=%s pose estimation failed for track=%s; skipping this crop",
                    self.camera.code,
                    track.track_id,
                )
                continue

            voter = self._gesture_voters.get(track.track_id)
            if voter is None:
                voter = GestureVoter(
                    min_frames_agreed=cfg.min_frames_agreed,
                    window_frames=cfg.window_frames,
                )
                self._gesture_voters[track.track_id] = voter

            candidate = None
            if pose is not None:
                # Model space -> this crop -> the original frame. crop_x/crop_y
                # exist on FrameTransform for exactly this "tile within the
                # frame" case, so the invariant that nothing downstream sees
                # model-space coordinates holds here too.
                mapped = replace(transform, crop_x=float(x1), crop_y=float(y1))
                pose = Pose(
                    tuple(
                        Keypoint(*mapped.point_to_original((kp.x, kp.y)), kp.conf)
                        for kp in pose.points
                    )
                )
                candidate = classify_gesture(
                    pose,
                    min_keypoint_conf=cfg.min_keypoint_conf,
                    raise_margin=cfg.raise_margin,
                    level_tolerance=cfg.level_tolerance,
                    extend_ratio=cfg.extend_ratio,
                    straight_tolerance=cfg.straight_tolerance,
                )

            settled = voter.add(candidate)
            if settled is None:
                # A real re-check that settled on nothing clears the held
                # state: this is how a lowered hand stops being HANDS_UP.
                self._gesture_last.pop(track.track_id, None)
                continue
            record = {
                "track_id": track.track_id,
                "code": settled.code,
                "conf": round(float(settled.conf), 3),
                "frames_agreed": int(settled.detail.get("frames_agreed", 0)),
            }
            self._gesture_last[track.track_id] = record
            if best is None or settled.conf > best[0]:
                best = (settled.conf, record)

        return best[1] if best else None

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
        health_publisher: Any = None,
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
        # Optional on purpose: the pipeline runs identically without it, and
        # every unit test builds one with no Redis in sight.
        self._health_publisher = health_publisher
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
            # These two int() truncations must match the ones FrameTransform.letterbox
            # used to derive pad_x/pad_y, or boxes come back off by a pixel.
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
            detections = self._to_detections(raws, transform, frame.width, frame.height)
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

    def _to_detections(
        self,
        raws: Sequence[Any],
        transform: FrameTransform,
        frame_w: float | None = None,
        frame_h: float | None = None,
    ) -> list[Detection]:
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
            if frame_w is not None and frame_h is not None:
                # A regression head can legitimately overshoot the letterboxed
                # canvas (the model has no notion of "edge of frame"); mapped
                # back through 1/scale that overshoot is magnified and can land
                # well outside the real frame. Nothing downstream clips it, so
                # an unclamped box lands off-canvas in the live overlay and
                # skews the foot-point zone/tripwire checks. This is the last
                # point that knows both the box and the original frame size.
                box = (
                    max(0.0, min(box[0], frame_w)),
                    max(0.0, min(box[1], frame_h)),
                    max(0.0, min(box[2], frame_w)),
                    max(0.0, min(box[3], frame_h)),
                )
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
            # Same snapshot the log line above summarises, handed to the API so
            # the dashboard can answer "are the cameras up?" without reading
            # this process's stdout. Never allowed to break the loop: telemetry
            # failing must not stop the stats thread that reports it.
            if self._health_publisher is not None:
                try:
                    self._health_publisher.publish(self.health())
                except Exception:
                    logger.debug("health snapshot publish failed", exc_info=True)

    def health(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.monotonic() - self._started_at, 1),
            "detector": self.detector.backend,
            "frames_dropped_queue": self.frame_queue.dropped,
            "pipeline": self.stats.as_dict(),
            "cameras": [w.health() for w in self.workers.values()],
        }
