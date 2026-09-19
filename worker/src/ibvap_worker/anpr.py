"""Automatic Number Plate Recognition (BUILD_SPEC §7.9).

Off the hot path: fed by crops from confirmed vehicle tracks, processed in a
small pool. Nothing here may block a frame.

Three ideas carry this module.

**Multi-frame voting is mandatory.** Single-frame OCR on a 100-px-wide plate
from a 2014 CCTV camera at dusk is a guess. The same normalised text has to win
``min_frames_agreed`` reads across the track before we believe it. P3: a wrong
plate on an alert is worse than no plate.

**Validation is structural, and correction is position-aware.** ``HR26DA1234``
has letters and digits in known places. ``0`` in a letter position is almost
certainly ``O``; ``O`` in a digit position is almost certainly ``0``. Correcting
blindly with a global map turns valid plates into invalid ones — the direction
of the substitution depends entirely on where the character sits.

**The database never holds a plate.** ``plate_hmac`` is what gets stored and
what gets matched against the watchlist. Plaintext exists only inside the
evidence document of an alert that actually fired, and reading it writes an
audit row (P6). A database leak must not be a list of who drove where.

The pure functions here (validation, normalisation, HMAC, voting) have no native
dependencies and are property-tested. The OCR backends are imported lazily.
"""

from __future__ import annotations

import hmac
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .types import BoxXYXY

__all__ = [
    "AnprConfig",
    "PlateCandidate",
    "PlateRead",
    "PlateVoter",
    "normalise_plate_text",
    "plate_hmac",
    "plate_matches",
    "to_storage_record",
    "validate_indian_plate",
    "vote",
]


@dataclass(frozen=True, slots=True)
class AnprConfig:
    """§7.9, read from ``config/anpr.yaml``. Everything below the read is a
    pure function; this is just the knobs for how pipeline.py drives them."""

    enabled: bool = True
    classes: tuple[str, ...] = ("vehicle",)
    min_frames_agreed: int = 3
    min_char_conf: float = 0.55
    window_frames: int = 30
    max_crops_per_track: int = 12
    region: str | None = "IN"
    hmac_key_env: str = "PLATE_HMAC_KEY"
    # "a single OCR pass is cheap next to detection" (pipeline.py's
    # _read_plates docstring) is true for the fine-tuned PaddleOCR-CTC ONNX
    # model this was designed around -- a few ms, resize+forward+decode. It is
    # not true of either fallback this repo actually ships with today
    # (models/anpr/ does not exist): RapidOCR runs a real text-DETECTION CNN
    # internally on every crop and measured 170-1700ms per call on this CPU,
    # which starved the shared detector thread and pushed its own inference
    # time from ~30ms to ~200ms with frames dropping every second. Gated the
    # same way weapon/gesture already gate their per-frame model calls; unlike
    # those, this is cheap to raise back to 1 once real plate-specific ONNX
    # weights are in place (see the class docstring above the fallback note).
    every_n_frames: int = 5
    # weapon/gesture both cap how many tracks get a model call in one frame
    # (max_tracks_per_frame); this had no such cap at all, so an eligible
    # frame with 5 vehicles in it fired 5 sequential OCR calls back to back --
    # with RapidOCR at 170-1700ms each, one frame could stall the stage
    # thread for several seconds. every_n_frames alone bounds frequency, not
    # worst-case burst size; this bounds the burst.
    max_tracks_per_frame: int = 2

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> AnprConfig:
        block = dict(cfg.get("anpr", cfg))
        voting = dict(block.get("voting", {}))
        storage = dict(block.get("storage", {}))
        return cls(
            enabled=bool(block.get("enabled", True)),
            classes=tuple(block.get("classes", ("vehicle",))),
            min_frames_agreed=int(voting.get("min_frames_agreed", 3)),
            min_char_conf=float(voting.get("min_char_conf", 0.55)),
            window_frames=int(voting.get("window_frames", 30)),
            max_crops_per_track=int(block.get("max_crops_per_track", 12)),
            region=block.get("region", "IN"),
            hmac_key_env=str(storage.get("hmac_key_env", "PLATE_HMAC_KEY")),
            every_n_frames=int(block.get("every_n_frames", 5)),
            max_tracks_per_frame=int(block.get("max_tracks_per_frame", 2)),
        )


# The five confusions that are ~80% of real CCTV OCR error (config/anpr.yaml).
# Read as: this letter is commonly emitted where that digit belongs.
LETTER_TO_DIGIT: dict[str, str] = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"}
DIGIT_TO_LETTER: dict[str, str] = {v: k for k, v in LETTER_TO_DIGIT.items()}

_STRIP_RE = re.compile(r"[^A-Z0-9]")

# Structural patterns, for the direct-match fast path (config/anpr.yaml).
PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"),  # HR26DA1234
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),  # 22BH1234AA (Bharat series)
    re.compile(r"^[A-Z]{2}[0-9]{2}[0-9]{4}$"),  # older / some military
)


@dataclass(frozen=True, slots=True)
class PlateRead:
    """One accepted plate, after voting."""

    text: str  # normalised, validated
    conf: float
    box: BoxXYXY
    region: str | None = None
    char_confs: tuple[float, ...] = ()
    frames_agreed: int = 0

    def hmac(self, key: bytes) -> str:
        return plate_hmac(self.text, key)


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    """One raw OCR result from one frame, before voting."""

    raw_text: str
    conf: float
    box: BoxXYXY
    char_confs: tuple[float, ...] = ()
    frame_id: int = 0


# ---------------------------------------------------------------------------
# Normalisation and validation
# ---------------------------------------------------------------------------


def _strip(text: str) -> str:
    """Uppercase and drop everything that is not A-Z0-9.

    Real OCR emits ``IND``, state emblems, hyphens, spaces and the odd bolt hole
    read as a full stop.
    """
    cleaned = _STRIP_RE.sub("", text.upper())
    # "IND" is stamped on the left of every modern Indian plate and is not part
    # of the registration number.
    if cleaned.startswith("IND") and len(cleaned) > 8:
        cleaned = cleaned[3:]
    return cleaned


def _templates_for_length(length: int) -> list[tuple[str, ...]]:
    """Character-class templates that could produce a string of this length.

    'A' = letter, 'N' = digit. Indian civilian format is
    ``<2 letters state><1-2 digits RTO><0-3 letters series><4 digits>``.
    """
    out: list[tuple[str, ...]] = []
    for rto in (1, 2):
        for series in (0, 1, 2, 3):
            if 2 + rto + series + 4 == length:
                out.append(tuple("AA" + "N" * rto + "A" * series + "NNNN"))
    # Bharat series: <2 digits year><BH><4 digits><1-2 letters>
    for suffix in (1, 2):
        if 2 + 2 + 4 + suffix == length:
            out.append(tuple("NN" + "BH" + "NNNN" + "A" * suffix))
    return out


def _coerce(text: str, template: Sequence[str]) -> str | None:
    """Force ``text`` into ``template``'s character classes using the confusion
    map. Returns None if any position needs a substitution we do not believe in.
    """
    if len(text) != len(template):
        return None
    out: list[str] = []
    for ch, want in zip(text, template, strict=True):
        if want == "A":
            if ch.isalpha():
                out.append(ch)
            elif ch in DIGIT_TO_LETTER:
                out.append(DIGIT_TO_LETTER[ch])
            else:
                return None
        elif want == "N":
            if ch.isdigit():
                out.append(ch)
            elif ch in LETTER_TO_DIGIT:
                out.append(LETTER_TO_DIGIT[ch])
            else:
                return None
        else:  # a literal, e.g. the 'B','H' of the Bharat template
            if ch == want:
                out.append(ch)
            elif want == "B" and ch == "8":
                out.append("B")
            elif want == "H":
                return None
            else:
                return None
    return "".join(out)


def normalise_plate_text(text: str) -> str:
    """Best-effort normalisation. Returns the corrected form, or the stripped
    input unchanged when no template fits.

    Idempotent: ``normalise(normalise(x)) == normalise(x)``, which the property
    tests assert — a non-idempotent normaliser makes voting compare apples to
    pears across frames.
    """
    stripped = _strip(text)
    if not stripped:
        return ""
    if any(p.match(stripped) for p in PATTERNS):
        return stripped
    for template in _templates_for_length(len(stripped)):
        coerced = _coerce(stripped, template)
        if coerced and any(p.match(coerced) for p in PATTERNS):
            return coerced
    return stripped


def validate_indian_plate(text: str) -> tuple[bool, str]:
    """Return ``(is_valid, normalised)``.

    Handles the O/0, I/1, S/5, B/8, Z/2 confusions position-aware, plus the
    standard, Bharat-series and older formats. Anything that does not fit a
    known structure is rejected — we would rather report no plate than a plate
    that does not exist.
    """
    if not isinstance(text, str):
        raise TypeError(f"plate text must be str, got {type(text).__name__}")
    normalised = normalise_plate_text(text)
    if not normalised:
        return False, ""
    if not 6 <= len(normalised) <= 11:
        return False, normalised
    return any(p.match(normalised) for p in PATTERNS), normalised


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def plate_hmac(text: str, key: bytes) -> str:
    """HMAC-SHA256 of the normalised plate. This is what the database stores.

    HMAC, not a bare hash: a plain SHA-256 of a plate is trivially reversible by
    enumerating the (small) plate space. The key lives in the environment, never
    in ``config/`` and never in git.
    """
    if not key:
        raise ValueError("plate HMAC key is empty; refusing to produce a guessable digest")
    if len(key) < 16:
        raise ValueError(f"plate HMAC key is {len(key)} bytes; need at least 16")
    _, normalised = validate_indian_plate(text)
    if not normalised:
        raise ValueError("cannot HMAC an empty plate")
    return hmac.new(key, normalised.encode("utf-8"), sha256).hexdigest()


def plate_matches(text: str, stored_hmac: str, key: bytes) -> bool:
    """Constant-time watchlist comparison."""
    return hmac.compare_digest(plate_hmac(text, key), stored_hmac)


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def vote(
    candidates: Sequence[PlateCandidate],
    *,
    min_frames_agreed: int = 3,
    min_char_conf: float = 0.55,
    region: str | None = "IN",
) -> PlateRead | None:
    """Accept a plate only when the same normalised text wins enough frames.

    Returns None when nothing reaches agreement — which is the common and
    correct outcome for a vehicle passing at 40 km/h in the rain.
    """
    if min_frames_agreed < 1:
        raise ValueError(f"min_frames_agreed must be >= 1, got {min_frames_agreed}")

    usable: list[tuple[str, PlateCandidate]] = []
    for cand in candidates:
        if cand.char_confs and min(cand.char_confs) < min_char_conf:
            continue  # one illegible character poisons the whole read
        valid, normalised = validate_indian_plate(cand.raw_text)
        if valid:
            usable.append((normalised, cand))

    if not usable:
        return None

    counts = Counter(text for text, _ in usable)
    winner, agreed = counts.most_common(1)[0]
    if agreed < min_frames_agreed:
        return None

    winning = [c for t, c in usable if t == winner]
    best = max(winning, key=lambda c: c.conf)
    mean_conf = sum(c.conf for c in winning) / len(winning)

    return PlateRead(
        text=winner,
        conf=round(mean_conf, 4),
        box=best.box,
        region=region,
        char_confs=best.char_confs,
        frames_agreed=agreed,
    )


@dataclass
class PlateVoter:
    """Accumulates candidates for one track and reports when a plate is settled.

    Stateful by necessity, but deliberately dumb: it holds candidates and defers
    every decision to the pure ``vote`` function above, so the logic stays
    testable without constructing a track.
    """

    min_frames_agreed: int = 3
    min_char_conf: float = 0.55
    window_frames: int = 30
    max_candidates: int = 12
    _candidates: list[PlateCandidate] = field(default_factory=list, repr=False)
    _settled: PlateRead | None = field(default=None, repr=False)

    def add(self, candidate: PlateCandidate) -> PlateRead | None:
        """Add a read; return the settled plate the first time it settles."""
        if self._settled is not None:
            return None
        self._candidates.append(candidate)
        if len(self._candidates) > self.max_candidates:
            self._candidates = self._candidates[-self.max_candidates :]
        result = vote(
            self._candidates,
            min_frames_agreed=self.min_frames_agreed,
            min_char_conf=self.min_char_conf,
        )
        if result is not None:
            self._settled = result
            return result
        return None

    @property
    def settled(self) -> PlateRead | None:
        return self._settled

    def reset(self) -> None:
        self._candidates.clear()
        self._settled = None


def to_storage_record(read: PlateRead, key: bytes, *, fired_alert: bool) -> dict[str, Any]:
    """Build the DB record for a plate read.

    ``fired_alert`` is the only thing that permits plaintext into the payload,
    and even then only into the alert's evidence document (P6, §7.9).
    """
    record: dict[str, Any] = {
        "plate_hmac": read.hmac(key),
        "conf": read.conf,
        "frames_agreed": read.frames_agreed,
        "region": read.region,
    }
    if fired_alert:
        record["plate_text"] = read.text
    return record


def build_reader(cfg: Mapping[str, Any]) -> Any:
    """Lazily construct the ONNX plate detector + recogniser (§7.9).

    Imported here rather than at module scope so that the pure functions above
    stay usable — and testable — with no native dependencies installed.
    """
    from .detect.plate_ocr import OnnxPlateReader

    return OnnxPlateReader(cfg)
