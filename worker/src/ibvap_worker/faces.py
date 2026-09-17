"""Face matching against a curated watchlist (BUILD_SPEC §7.10). **OPT-IN**

This module is the answer to the privacy question, and the shape of it is set
by P6 rather than by what is technically convenient:

* ``faces.enabled`` defaults to **false**, and nothing here loads a weight file
  while it is. Turning it on is an admin action that writes an audit row.
* There is **no enrolment path here at all.** A face can only enter the
  watchlist through the audited admin API. The worker reads; it never writes.
* An embedding computed from a passer-by who matches nobody is held in memory
  for the life of that track and destroyed with it. It is never written to
  disk, never logged, never attached to an alert, and never sent to a sink.
  ``retain_non_matching_embeddings: false`` is not a tuning knob; it is the
  invariant that makes this a *watchlist* system rather than a surveillance
  dragnet.

That last point is the one worth defending out loud: the difference between
"we check faces against a list a human curated and audited" and "we build a
biometric record of everyone who walks past" is exactly whether non-matching
embeddings are kept. This module keeps none.

Pure: no I/O, no ONNX, no numpy at import time. The backends live in
detect/onnx_face.py, so the matching logic here stays testable without weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FaceConfig",
    "FaceDetection",
    "FaceMatch",
    "FaceSettled",
    "FaceVoter",
    "FaceWatchlist",
    "WatchlistFace",
    "cosine_similarity",
    "vote_face",
]


@dataclass(frozen=True, slots=True)
class FaceConfig:
    """The ``faces:`` block (config/faces.yaml)."""

    enabled: bool = False
    detector_weights: str = "models/face/scrfd_500m.onnx"
    embedder_weights: str = "models/face/w600k_mbf.onnx"
    #: SCRFD's own input size, distinct from ArcFace's fixed 112x112 (the
    #: embedder's input size is not configurable -- it is what the model was
    #: trained on, and align_face always produces exactly that).
    detector_input_size: tuple[int, int] = (640, 640)
    embedding_dim: int = 512
    classes: tuple[str, ...] = ("person",)
    #: Below this a face is too few pixels for an embedding to mean anything.
    #: Matching a 20-px face is not recognition, it is a coin toss with a
    #: person's name attached.
    min_face_px: int = 40
    detect_conf: float = 0.60
    #: Cosine similarity. Deliberately strict: a false positive here accuses
    #: a specific, named human being (P3, P6).
    match_threshold: float = 0.55
    max_faces_per_frame: int = 10
    #: THE INVARIANT. See the module docstring.
    retain_non_matching_embeddings: bool = False
    enrolment_requires_audit: bool = True
    every_n_frames: int = 3
    max_tracks_per_frame: int = 2
    min_frames_agreed: int = 2
    window_frames: int = 6
    #: Cheap "has this crop changed" gate in front of the model (activity.py).
    activity_gate: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> FaceConfig:
        block = dict(cfg.get("faces", cfg))
        voting = dict(block.get("voting", {}) or {})
        det_size = list(block.get("detector_input_size", (640, 640)))
        return cls(
            enabled=bool(block.get("enabled", False)),
            detector_weights=str(block.get("detector_weights", "models/face/scrfd_500m.onnx")),
            embedder_weights=str(block.get("embedder_weights", "models/face/w600k_mbf.onnx")),
            detector_input_size=(int(det_size[0]), int(det_size[1])),
            embedding_dim=int(block.get("embedding_dim", 512)),
            classes=tuple(block.get("classes", ("person",))),
            min_face_px=int(block.get("min_face_px", 40)),
            detect_conf=float(block.get("detect_conf", 0.60)),
            match_threshold=float(block.get("match_threshold", 0.55)),
            max_faces_per_frame=int(block.get("max_faces_per_frame", 10)),
            retain_non_matching_embeddings=bool(block.get("retain_non_matching_embeddings", False)),
            enrolment_requires_audit=bool(block.get("enrolment_requires_audit", True)),
            every_n_frames=max(1, int(block.get("every_n_frames", 3))),
            max_tracks_per_frame=int(block.get("max_tracks_per_frame", 2)),
            min_frames_agreed=int(voting.get("min_frames_agreed", 2)),
            window_frames=int(voting.get("window_frames", 6)),
            activity_gate=dict(block.get("activity_gate", {}) or {}),
        )


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """One detected face in ORIGINAL frame coordinates.

    ``landmarks`` are the five ArcFace points (eyes, nose, mouth corners) and
    are what the embedder aligns on -- an unaligned crop embeds poorly enough
    to turn a real match into a miss.
    """

    box: tuple[float, float, float, float]
    score: float
    landmarks: tuple[tuple[float, float], ...] = ()

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])


@dataclass(frozen=True, slots=True)
class WatchlistFace:
    """One enrolled face. Reaches the worker only via the audited admin API."""

    person_id: str
    ref_code: str  # the operator-facing reference, never a name in logs
    category: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FaceMatch:
    person_id: str
    ref_code: str
    category: str
    similarity: float


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two embeddings, in [-1, 1].

    Computed rather than assumed-normalised: an embedder that returns
    unnormalised vectors would otherwise silently produce similarities above 1
    and match everybody. Raises on a length mismatch, because comparing a
    512-d embedding with a 128-d one is a configuration bug that must not be
    allowed to return a plausible-looking number.
    """
    if len(a) != len(b):
        raise ValueError(f"embedding dimensions differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


@dataclass
class FaceWatchlist:
    """In-memory index of enrolled faces. Read-only from the worker's side.

    Deliberately a plain linear scan rather than the database's HNSW index: a
    curated watchlist at a border post is tens of people, not millions, and at
    that size an exact scan is both faster than a round trip and immune to the
    approximate-index failure mode where a real match is simply not returned.
    The pgvector index exists for the API's enrolment-time queries.
    """

    threshold: float = 0.55
    _faces: list[WatchlistFace] = field(default_factory=list, repr=False)

    def set_faces(self, faces: Sequence[WatchlistFace]) -> None:
        self._faces = list(faces)

    @property
    def count(self) -> int:
        return len(self._faces)

    def match(self, embedding: Sequence[float]) -> FaceMatch | None:
        """Best match above the threshold, or None.

        Returning only the single best is deliberate: handing an operator a
        ranked list of people a face *might* be invites them to pick the one
        that fits their theory. One answer or none.
        """
        best: FaceMatch | None = None
        for face in self._faces:
            similarity = cosine_similarity(embedding, face.embedding)
            if similarity < self.threshold:
                continue
            if best is None or similarity > best.similarity:
                best = FaceMatch(
                    person_id=face.person_id,
                    ref_code=face.ref_code,
                    category=face.category,
                    similarity=round(similarity, 4),
                )
        return best


@dataclass(frozen=True, slots=True)
class FaceSettled:
    """A watchlist match confirmed across enough frames to name in an alert."""

    person_id: str
    ref_code: str
    category: str
    similarity: float  # mean of the agreeing frames
    frames_agreed: int


def vote_face(
    candidates: Sequence[FaceMatch | None], *, min_frames_agreed: int = 2
) -> FaceSettled | None:
    """Settle on a person only once enough recent frames agree on WHO.

    Naming a specific human being is the highest-stakes false positive this
    system can produce (P3, P6), so this needs the strongest agreement of any
    voter here: the plurality must be on the same ``person_id``, not merely on
    "matched somebody" the way weapon.py votes on "armed" -- two different
    watchlist entries drawing similar scores must never average into a
    confident-looking answer that names neither of them correctly.
    """
    if min_frames_agreed < 1:
        raise ValueError(f"min_frames_agreed must be positive, got {min_frames_agreed}")

    seen = [c for c in candidates if c is not None]
    if len(seen) < min_frames_agreed:
        return None

    by_person: dict[str, list[FaceMatch]] = {}
    for c in seen:
        by_person.setdefault(c.person_id, []).append(c)
    person_id, matches = max(
        by_person.items(), key=lambda kv: (len(kv[1]), sum(m.similarity for m in kv[1]))
    )
    if len(matches) < min_frames_agreed:
        return None

    first = matches[0]
    return FaceSettled(
        person_id=person_id,
        ref_code=first.ref_code,
        category=first.category,
        similarity=round(sum(m.similarity for m in matches) / len(matches), 4),
        frames_agreed=len(matches),
    )


@dataclass
class FaceVoter:
    """Rolling per-track window (mirrors GestureVoter/WeaponVoter).

    Never latches: this reports what the recent window says every time it is
    asked, so a track that walks past a camera at an angle that stops
    resembling anyone stops being reported as a match.
    """

    min_frames_agreed: int = 2
    window_frames: int = 6
    _candidates: list[FaceMatch | None] = field(default_factory=list, repr=False)

    def add(self, candidate: FaceMatch | None) -> FaceSettled | None:
        self._candidates.append(candidate)
        if len(self._candidates) > self.window_frames:
            self._candidates = self._candidates[-self.window_frames :]
        return vote_face(self._candidates, min_frames_agreed=self.min_frames_agreed)

    def reset(self) -> None:
        self._candidates.clear()
