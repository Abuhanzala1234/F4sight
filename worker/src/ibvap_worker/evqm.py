"""Environmental & Video Quality Monitor (BUILD_SPEC §7.2).

EVQM answers one question: *what is the scene doing to the image right now?*
The answer selects a processing profile — ``day``, ``lowlight``, ``night``,
``fog``, ``degraded`` — which drives enhancement (§7.3) and softens some rules.

Blocker #5, quoting CLAUDE.md:

    EVQM flapping. Without hysteresis, a passing headlight flips the processing
    profile every second. enter_samples=3, exit_samples=5.

Hysteresis is therefore not a refinement, it is the feature. A candidate profile
must win three consecutive votes before it is entered, and the incumbent must
lose five consecutive votes before it is left. Entering is faster than leaving
on purpose: reacting quickly to nightfall is useful, reacting quickly to a truck's
headlights is not.

Budget: under 3 ms on a 320-px downscale, sampled every 15th frame. EVQM is not
allowed to be interesting.

The vote logic — ``QualityMetrics.profile_vote`` and ``ProfileHysteresis`` — is
pure and testable without numpy. Only ``EVQM.observe`` touches an image.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import QualityMetrics

if TYPE_CHECKING:  # pragma: no cover

    from .types import Frame

__all__ = ["EVQM", "EVQMConfig", "Profile", "ProfileHysteresis", "profile_vote"]

DAY = "day"
LOWLIGHT = "lowlight"
NIGHT = "night"
FOG = "fog"
DEGRADED = "degraded"
ALL_PROFILES = (DAY, LOWLIGHT, NIGHT, FOG, DEGRADED)

Profile = str


@dataclass(frozen=True, slots=True)
class EVQMConfig:
    enabled: bool = True
    sample_every_n: int = 15
    downscale_width: int = 320
    enter_samples: int = 3
    exit_samples: int = 5

    night_brightness_below: float = 0.18
    lowlight_brightness_below: float = 0.35
    degraded_contrast_below: float = 0.10
    degraded_blur_below: float = 0.12
    fog_above: float = 0.55
    degraded_noise_above: float = 0.45

    laplacian_var_sharp: float = 500.0
    noise_energy_max: float = 40.0
    motion_pixel_delta: int = 25

    tamper_enabled: bool = True
    tamper_brightness_step: float = 0.35
    tamper_blur_step: float = 0.30
    tamper_min_samples: int = 20

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> EVQMConfig:
        block = dict(cfg.get("evqm", cfg))
        hyst = dict(block.get("hysteresis", {}))
        thr = dict(block.get("thresholds", {}))
        norm = dict(block.get("normalisation", {}))
        tamper = dict(block.get("tamper", {}))
        return cls(
            enabled=bool(block.get("enabled", True)),
            sample_every_n=int(block.get("sample_every_n", 15)),
            downscale_width=int(block.get("downscale_width", 320)),
            enter_samples=int(hyst.get("enter_samples", 3)),
            exit_samples=int(hyst.get("exit_samples", 5)),
            night_brightness_below=float(thr.get("night_brightness_below", 0.18)),
            lowlight_brightness_below=float(thr.get("lowlight_brightness_below", 0.35)),
            degraded_contrast_below=float(thr.get("degraded_contrast_below", 0.10)),
            degraded_blur_below=float(thr.get("degraded_blur_below", 0.12)),
            fog_above=float(thr.get("fog_above", 0.55)),
            degraded_noise_above=float(thr.get("degraded_noise_above", 0.45)),
            laplacian_var_sharp=float(norm.get("laplacian_var_sharp", 500.0)),
            noise_energy_max=float(norm.get("noise_energy_max", 40.0)),
            motion_pixel_delta=int(norm.get("motion_pixel_delta", 25)),
            tamper_enabled=bool(tamper.get("enabled", True)),
            tamper_brightness_step=float(tamper.get("brightness_step", 0.35)),
            tamper_blur_step=float(tamper.get("blur_step", 0.30)),
            tamper_min_samples=int(tamper.get("min_samples_before_arming", 20)),
        )


# ---------------------------------------------------------------------------
# Voting (pure)
# ---------------------------------------------------------------------------


def profile_vote(m: QualityMetrics, cfg: EVQMConfig) -> Profile:
    """One sample's opinion about the processing profile.

    Order matters. Fog is checked before darkness because fog at night is still
    fog — dehazing a hazy night scene helps, and a CLAHE-only night profile does
    not. Structural degradation (blur, noise, no contrast) outranks both,
    because when the image itself is broken no enhancement will rescue it and
    the risk model should be told to discount what it sees.
    """
    if m.blur < cfg.degraded_blur_below or m.noise > cfg.degraded_noise_above:
        return DEGRADED
    if m.contrast < cfg.degraded_contrast_below and m.fog <= cfg.fog_above:
        return DEGRADED
    if m.fog > cfg.fog_above:
        return FOG
    if m.brightness < cfg.night_brightness_below:
        return NIGHT
    if m.brightness < cfg.lowlight_brightness_below:
        return LOWLIGHT
    return DAY


@dataclass
class ProfileHysteresis:
    """The anti-flap state machine. Pure, deterministic, trivially testable.

    Two counters, not one: ``_enter_streak`` tracks consecutive votes for a
    single challenger, ``_exit_streak`` tracks consecutive votes against the
    incumbent. A challenger that is merely *different each time* (a headlight
    making one sample look like DAY, the next like DEGRADED) never accumulates
    an entry streak, so the profile holds — which is exactly the behaviour
    blocker #5 asks for.
    """

    enter_samples: int = 3
    exit_samples: int = 5
    current: Profile = DAY
    _candidate: Profile | None = field(default=None, repr=False)
    _enter_streak: int = field(default=0, repr=False)
    _exit_streak: int = field(default=0, repr=False)
    changes: int = 0

    def __post_init__(self) -> None:
        if self.enter_samples < 1 or self.exit_samples < 1:
            raise ValueError(
                f"hysteresis counts must be >= 1 "
                f"(enter={self.enter_samples}, exit={self.exit_samples})"
            )

    def submit(self, vote: Profile) -> Profile:
        """Feed one vote; return the (possibly unchanged) active profile."""
        if vote not in ALL_PROFILES:
            raise ValueError(f"unknown profile vote {vote!r}; expected one of {ALL_PROFILES}")

        if vote == self.current:
            # The incumbent was reaffirmed; both streaks reset.
            self._exit_streak = 0
            self._candidate = None
            self._enter_streak = 0
            return self.current

        self._exit_streak += 1
        if vote == self._candidate:
            self._enter_streak += 1
        else:
            self._candidate = vote
            self._enter_streak = 1

        # BOTH conditions must hold: the challenger has earned its place AND the
        # incumbent has lost enough ground. Requiring only the first would let a
        # three-frame headlight sweep win.
        if self._enter_streak >= self.enter_samples and self._exit_streak >= self.exit_samples:
            self.current = vote
            self.changes += 1
            self._candidate = None
            self._enter_streak = 0
            self._exit_streak = 0
        return self.current


# ---------------------------------------------------------------------------
# Metric extraction (needs numpy + cv2)
# ---------------------------------------------------------------------------


class EVQM:
    """Samples frames, computes metrics, and owns the profile state machine."""

    def __init__(self, cfg: EVQMConfig, camera_id: str = "") -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self._hyst = ProfileHysteresis(
            enter_samples=cfg.enter_samples,
            exit_samples=cfg.exit_samples,
        )
        self._frame_counter = 0
        self._sample_count = 0
        self._metrics: QualityMetrics | None = None
        self._prev_gray: Any = None
        self._prev_metrics: QualityMetrics | None = None
        self._tamper_flag = False

    @property
    def profile(self) -> Profile:
        return self._hyst.current

    @property
    def metrics(self) -> QualityMetrics | None:
        return self._metrics

    @property
    def tamper_suspected(self) -> bool:
        return self._tamper_flag

    @property
    def profile_changes(self) -> int:
        return self._hyst.changes

    def observe(self, frame: Frame) -> QualityMetrics | None:
        """Sample this frame if it is due; return metrics, or None if skipped."""
        if not self.cfg.enabled:
            return None
        self._frame_counter += 1
        if self._frame_counter % self.cfg.sample_every_n != 0:
            return None

        metrics = self._compute(frame.image)
        self._sample_count += 1
        self._detect_tamper(metrics)
        self._prev_metrics = metrics
        self._metrics = metrics
        self._hyst.submit(profile_vote(metrics, self.cfg))
        return metrics

    # -- internals ---------------------------------------------------------

    def _compute(self, image: Any) -> QualityMetrics:
        import cv2
        import numpy as np

        h, w = image.shape[:2]
        if w > self.cfg.downscale_width:
            scale = self.cfg.downscale_width / w
            small = cv2.resize(
                image,
                (self.cfg.downscale_width, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        brightness = float(gray_f.mean() / 255.0)
        contrast = float(gray_f.std() / 128.0)

        # Variance of Laplacian: the standard focus measure. Normalised so that
        # 1.0 means "as sharp as we ever expect", which is a tuning constant and
        # therefore lives in config.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur = min(1.0, lap_var / self.cfg.laplacian_var_sharp)

        # Dark channel prior: haze lifts the per-pixel channel minimum off zero
        # across the whole image. A clear scene almost always has some pixel in
        # some channel near black within any 15-px window.
        min_channel = small.min(axis=2)
        dark = cv2.erode(min_channel, np.ones((15, 15), np.uint8))
        fog = float(dark.mean() / 255.0)

        # Noise: residual energy after a median filter, which removes structure
        # but not sensor grain.
        denoised = cv2.medianBlur(gray, 3)
        noise_energy = float(np.abs(gray_f - denoised.astype(np.float32)).mean())
        noise = min(1.0, noise_energy / self.cfg.noise_energy_max)

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            motion = float((diff > self.cfg.motion_pixel_delta).mean())
        else:
            motion = 0.0
        self._prev_gray = gray

        return QualityMetrics(
            brightness=brightness,
            contrast=min(1.0, contrast),
            blur=blur,
            fog=fog,
            noise=noise,
            motion=motion,
        )

    def _detect_tamper(self, metrics: QualityMetrics) -> None:
        """A step change the sun cannot explain (§7.7 CAMERA_TAMPER).

        Sunset is gradual; a lens cover, a spray can or a rotated mount is not.
        We arm only after enough samples so that startup transients do not fire
        it, and we look for a step between consecutive samples rather than an
        absolute level.
        """
        self._tamper_flag = False
        if not self.cfg.tamper_enabled:
            return
        if self._sample_count < self.cfg.tamper_min_samples or self._prev_metrics is None:
            return
        d_bright = abs(metrics.brightness - self._prev_metrics.brightness)
        d_blur = abs(metrics.blur - self._prev_metrics.blur)
        self._tamper_flag = (
            d_bright > self.cfg.tamper_brightness_step or d_blur > self.cfg.tamper_blur_step
        )
