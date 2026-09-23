"""Pydantic v2 schemas — everything crossing the process boundary (§8).

Note what is *not* here: no schema exposes ``Camera.rtsp_url`` (it holds
credentials) and no schema exposes a plate in plaintext except
``PlateRevealOut``, which is only reachable by an investigator and writes an
audit row on the way (P6, §12).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: An alert id in a URL path. Validated here so a malformed one (the dashboard
#: once sent the literal string "undefined") is a 422 at the edge rather than
#: a Postgres DataError surfacing as a 500.
AlertId = Annotated[
    str,
    Path(
        pattern=r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
    ),
]

Severity = Literal["info", "low", "medium", "high", "critical"]
AlertStatus = Literal["raised", "acknowledged", "adjudicated"]
Adjudication = Literal["true_positive", "false_positive", "unclear"]
Role = Literal["viewer", "operator", "investigator", "admin"]
Verdict = Literal["VERIFIED", "PENDING_ANCHOR", "TAMPERED", "UNVERIFIABLE"]
ZoneKind = Literal["area", "tripwire", "mask"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    role: Role
    display_name: str


class UserOut(ORMModel):
    id: str
    username: str
    display_name: str
    role: Role
    active: bool


# --------------------------------------------------------------------------
# Sites, cameras, zones
# --------------------------------------------------------------------------


class SiteOut(ORMModel):
    id: str
    code: str
    name: str
    sector: str | None = None
    lat: float | None = None
    lon: float | None = None
    timezone: str


class CameraOut(ORMModel):
    """Deliberately omits rtsp_url — it carries camera credentials (§12)."""

    id: str
    site_id: str
    code: str
    name: str
    mediamtx_path: str
    resolution_w: int
    resolution_h: int
    native_fps: float
    analytics_fps: float
    is_recording_only: bool
    enabled: bool


class StreamOut(BaseModel):
    camera_id: str
    mediamtx_path: str
    hls_url: str
    webrtc_url: str
    # Browsers cannot play RTSP (blocker #4). Saying so in the payload is
    # cheaper than the conversation it prevents.
    note: str = "RTSP is not playable in a browser; use hls_url."


class CameraConnectIn(BaseModel):
    """Bind an existing camera slot to a live IP camera (e.g. a phone running
    the 'IP Webcam' app). No model/detector step involved — this only wires
    up the video source; the detector is already running and shared across
    every camera."""

    ip: str = Field(min_length=1, max_length=253)
    port: int = Field(default=8080, ge=1, le=65535)
    path: str = Field(default="h264_ulaw.sdp", max_length=128)

    @field_validator("ip")
    @classmethod
    def _no_embedded_port(cls, value: str) -> str:
        """A pasted `192.168.1.12:8080` (people copy the IP Webcam app's own
        on-screen address, port included) must not reach connect_camera as-is:
        the port field defaults to 8080 too, so the RTSP source built from
        both becomes `192.168.1.12:8080:8080` -- a hostname MediaMTX can't
        resolve, which fails as a silent 'no signal' tile with no obvious
        cause. Reject it here with a message pointing at the actual mistake,
        rather than letting a broken URL through to MediaMTX."""
        host = value.strip()
        if host.count(":") == 1 and not host.startswith("["):
            raise ValueError(
                f"'{host}' looks like host:port -- put only the IP address here "
                "and the port in the Port field below."
            )
        return host


class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    kind: ZoneKind
    polygon: list[tuple[float, float]]
    direction: Literal["in", "out", "both"] | None = None
    classes: list[str] = Field(default_factory=list)
    schedule: dict[str, Any] | None = None
    severity_base: int = Field(default=3, ge=1, le=5)
    enabled: bool = True

    @field_validator("polygon")
    @classmethod
    def _normalised(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Polygons are stored normalised so a resolution change does not
        invalidate every zone an operator drew (§6.2)."""
        for x, y in value:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"polygon points must be normalised to 0..1, got ({x}, {y}). "
                    "Divide pixel coordinates by the camera's resolution."
                )
        return value

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> ZoneIn:
        if self.kind == "tripwire" and len(self.polygon) != 2:
            raise ValueError(
                f"a tripwire is exactly 2 points, got {len(self.polygon)}. "
                "Point order sets which side is 'in'."
            )
        if self.kind in ("area", "mask") and len(self.polygon) < 3:
            raise ValueError(f"an {self.kind} zone needs at least 3 points")
        return self


class ZoneOut(ORMModel):
    id: str
    camera_id: str
    name: str
    kind: ZoneKind
    polygon: list[list[float]]
    direction: str | None
    classes: list[str] | None
    schedule: dict[str, Any] | None
    severity_base: int
    enabled: bool


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------


class RiskContribution(BaseModel):
    code: str
    weight: float
    detail: dict[str, Any] = Field(default_factory=dict)


class AlertSummary(ORMModel):
    id: str
    site_id: str
    camera_id: str
    kind: str
    severity: Severity
    risk_score: float
    reason_codes: list[str]
    ts_utc: datetime
    status: AlertStatus
    adjudication: Adjudication | None = None
    ledger_status: str


class EvidenceItemOut(ORMModel):
    id: str
    kind: str
    sha256: str
    bytes: int | None
    width: int | None
    height: int | None
    captured_at: datetime | None
    enhanced: bool
    enhancement_params: dict[str, Any] | None


class AlertDetail(AlertSummary):
    track_id: int | None
    zone_id: str | None
    window_start: datetime | None
    window_end: datetime | None
    risk_breakdown: list[RiskContribution]
    evidence_hash: str
    evidence_doc: dict[str, Any]
    items: list[EvidenceItemOut] = Field(default_factory=list)

    @model_validator(mode="after")
    def _breakdown_sums_to_score(self) -> AlertDetail:
        """P2, checked one last time on the way out of the door.

        If the stored breakdown does not add up, the explanation we are about to
        show an operator is a lie. Better a 500 than a confident wrong number.
        """
        total = round(sum(c.weight for c in self.risk_breakdown), 2)
        if abs(total - float(self.risk_score)) > 0.01:
            raise ValueError(
                f"risk breakdown does not sum to risk_score for alert {self.id}: "
                f"sum={total} score={self.risk_score}"
            )
        return self


class AlertPage(BaseModel):
    items: list[AlertSummary]
    next_cursor: str | None = None
    total_estimate: int | None = None


class AdjudicateIn(BaseModel):
    verdict: Adjudication
    note: str | None = Field(default=None, max_length=2000)


# --------------------------------------------------------------------------
# Verification — the payload that wins the blockchain argument (§8)
# --------------------------------------------------------------------------


class VerificationCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class MerkleProofOut(BaseModel):
    leaf_index: int
    leaf_hash: str
    proof: list[tuple[str, str]]
    computed_root: str
    stored_root: str
    root_match: bool


class LedgerInfoOut(BaseModel):
    backend: str
    tx_id: str | None = None
    block_number: int | None = None
    anchored_at: datetime | None = None
    root_on_chain: str | None = None
    chain_match: bool = False


class EvidenceItemVerification(BaseModel):
    kind: str
    object_key: str
    sha256: str
    file_matches: bool | None = None  # None when the object store is unreachable


class VerificationOut(BaseModel):
    alert_id: str
    stored_hash: str
    recomputed_hash: str
    hash_match: bool
    canonical_bytes_sha256: str
    canonical_length: int
    evidence_items: list[EvidenceItemVerification] = Field(default_factory=list)
    merkle: MerkleProofOut | None = None
    ledger: LedgerInfoOut | None = None
    verdict: Verdict
    checks: list[VerificationCheck] = Field(default_factory=list)
    diff: list[str] = Field(default_factory=list)


class DocumentVerifyIn(BaseModel):
    """Verify an evidence JSON somebody hands you — the 'prove it to a sceptic'
    endpoint. Works without the alert existing in this database."""

    document: dict[str, Any]
    expected_hash: str | None = None


# --------------------------------------------------------------------------
# Watchlists
# --------------------------------------------------------------------------


class VehicleWatchIn(BaseModel):
    """Plate arrives in plaintext and is HMAC'd before storage. It is never
    echoed back (P6)."""

    plate: str = Field(min_length=4, max_length=16)
    category: str = "watch"
    region: str = "IN"
    expires_at: datetime | None = None


class VehicleWatchOut(ORMModel):
    id: str
    plate_hmac: str
    region: str
    category: str
    active: bool
    expires_at: datetime | None


class PersonWatchIn(BaseModel):
    ref_code: str = Field(min_length=1, max_length=64)
    display_name: str | None = None
    category: str = "person_of_interest"
    expires_at: datetime | None = None


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    latency_ms: float | None = None


class HealthOut(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    components: list[ComponentHealth]
    cameras: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
