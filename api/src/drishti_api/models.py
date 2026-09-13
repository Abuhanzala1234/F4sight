"""SQLAlchemy ORM — the §6 data model.

Several of the project's invariants are enforced here as database constraints,
not merely in Python. That is deliberate: a constraint in application code
protects the code path you remembered, and a constraint in the database protects
every path, including the psql session someone opens at 2 a.m.

The three that matter most:

* ``evidence_item`` snapshots and clips cannot be marked ``enhanced`` (P4);
* ``alert.risk_score`` must lie in [0, 100] and carry a non-empty breakdown (P2);
* ``watchlist_vehicle`` has a ``plate_hmac`` column and no plaintext column at
  all — you cannot leak a field that does not exist (P6).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# UUIDv7 from infra/postgres/init.sql: time-sortable, which matters for evidence
# ordering. A random UUID makes "what happened next" an index-less sort.
UUID7 = text("uuid7()")


def _pk() -> Mapped[str]:
    return mapped_column(UUID(as_uuid=False), primary_key=True, server_default=UUID7)


def _now() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Site(Base):
    __tablename__ = "site"

    id: Mapped[str] = _pk()
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(64))
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    timezone: Mapped[str] = mapped_column(
        String(64), default="Asia/Kolkata", nullable=False
    )
    created_at: Mapped[datetime] = _now()

    cameras: Mapped[list[Camera]] = relationship(back_populates="site")


class Camera(Base):
    __tablename__ = "camera"
    __table_args__ = (
        UniqueConstraint("site_id", "code", name="uq_camera_site_code"),
        CheckConstraint("analytics_fps > 0", name="ck_camera_fps_positive"),
    )

    id: Mapped[str] = _pk()
    site_id: Mapped[str] = mapped_column(
        ForeignKey("site.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # Encrypted at rest (§12). The API never returns this field.
    rtsp_url: Mapped[str | None] = mapped_column(Text)
    mediamtx_path: Mapped[str] = mapped_column(String(128), nullable=False)
    resolution_w: Mapped[int] = mapped_column(Integer, default=1280)
    resolution_h: Mapped[int] = mapped_column(Integer, default=720)
    native_fps: Mapped[float] = mapped_column(Float, default=25.0)
    analytics_fps: Mapped[float] = mapped_column(Float, default=6.0)
    # P8's escape hatch: MediaMTX still records it, the worker ignores it.
    is_recording_only: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    profile: Mapped[str] = mapped_column(String(16), default="laptop")
    lens: Mapped[str] = mapped_column(String(16), default="fixed")
    calibration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = _now()

    site: Mapped[Site] = relationship(back_populates="cameras")
    zones: Mapped[list[Zone]] = relationship(back_populates="camera")


class Zone(Base):
    __tablename__ = "zone"
    __table_args__ = (
        CheckConstraint("kind IN ('area','tripwire','mask')", name="ck_zone_kind"),
        CheckConstraint(
            "direction IS NULL OR direction IN ('in','out','both')",
            name="ck_zone_direction",
        ),
        CheckConstraint("severity_base BETWEEN 1 AND 5", name="ck_zone_severity"),
        Index("ix_zone_camera", "camera_id"),
    )

    id: Mapped[str] = _pk()
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # Normalised 0..1 coordinates, so changing a camera's resolution does not
    # invalidate every polygon an operator drew (§6.2).
    polygon: Mapped[list[list[float]]] = mapped_column(JSONB, nullable=False)
    direction: Mapped[str | None] = mapped_column(String(8))
    classes: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    schedule: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    severity_base: Mapped[int] = mapped_column(SmallInteger, default=3)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = _now()

    camera: Mapped[Camera] = relationship(back_populates="zones")


class Track(Base):
    __tablename__ = "track"
    __table_args__ = (Index("ix_track_camera_time", "camera_id", "first_seen_at"),)

    id: Mapped[str] = _pk()
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id", ondelete="CASCADE"))
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    cls: Mapped[str] = mapped_column(String(32), nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    max_conf: Mapped[float] = mapped_column(Float, default=0.0)
    path: Mapped[list[Any] | None] = mapped_column(JSONB)
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Event(Base):
    """A rule firing, pre-debounce. Cheap, high-volume, retained 7 days."""

    __tablename__ = "event"
    __table_args__ = (Index("ix_event_camera_time", "camera_id", "ts_utc"),)

    id: Mapped[str] = _pk()
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id", ondelete="CASCADE"))
    track_id: Mapped[str | None] = mapped_column(
        ForeignKey("track.id", ondelete="SET NULL")
    )
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("zone.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class Alert(Base):
    """The debounced, scored, operator-visible thing. [DEMO-CRITICAL]"""

    __tablename__ = "alert"
    __table_args__ = (
        CheckConstraint(
            "risk_score >= 0 AND risk_score <= 100", name="ck_alert_risk_range"
        ),
        # P2: an alert that cannot be explained must not exist. An empty
        # breakdown is an unexplainable alert, so the database refuses it.
        CheckConstraint(
            "jsonb_array_length(risk_breakdown) > 0", name="ck_alert_has_breakdown"
        ),
        CheckConstraint(
            "severity IN ('info','low','medium','high','critical')",
            name="ck_alert_severity",
        ),
        CheckConstraint(
            "status IN ('raised','acknowledged','adjudicated')", name="ck_alert_status"
        ),
        CheckConstraint(
            "adjudication IS NULL OR adjudication IN "
            "('true_positive','false_positive','unclear')",
            name="ck_alert_adjudication",
        ),
        CheckConstraint(
            "ledger_status IN ('pending','anchored','failed')",
            name="ck_alert_ledger_status",
        ),
        CheckConstraint("char_length(evidence_hash) = 64", name="ck_alert_hash_length"),
        # Keyset pagination on (ts_utc, id) — see §8.
        Index("ix_alert_feed", text("ts_utc DESC"), text("id DESC")),
        Index("ix_alert_site_time", "site_id", text("ts_utc DESC")),
        Index("ix_alert_status", "status"),
        Index(
            "ix_alert_ledger_pending",
            "ledger_status",
            postgresql_where=text("ledger_status = 'pending'"),
        ),
    )

    id: Mapped[str] = _pk()
    site_id: Mapped[str] = mapped_column(
        ForeignKey("site.id", ondelete="CASCADE"), index=True
    )
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id", ondelete="CASCADE"))
    track_id: Mapped[int | None] = mapped_column(Integer)
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("zone.id", ondelete="SET NULL")
    )
    primary_event_id: Mapped[str | None] = mapped_column(
        ForeignKey("event.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    risk_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    # INVARIANT (P2): the weights in here sum exactly to risk_score. Verified in
    # risk.py, by property test, and again by the API before it serves the row.
    risk_breakdown: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    reason_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="raised", nullable=False)
    adjudication: Mapped[str | None] = mapped_column(String(16))
    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # The exact object that was hashed, so verification is a pure function of
    # one row (§6.2).
    evidence_doc: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    ledger_status: Mapped[str] = mapped_column(
        String(16), default="pending", nullable=False
    )
    created_at: Mapped[datetime] = _now()

    items: Mapped[list[EvidenceItem]] = relationship(
        back_populates="alert", cascade="all, delete-orphan"
    )


class EvidenceItem(Base):
    __tablename__ = "evidence_item"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('snapshot','clip','crop','plate_crop')", name="ck_evidence_kind"
        ),
        # P4, enforced by the database. Evidence is for the court; enhancement
        # is for the model. A snapshot marked enhanced cannot be inserted.
        CheckConstraint(
            "NOT (kind IN ('snapshot','clip') AND enhanced)",
            name="ck_evidence_unenhanced",
        ),
        CheckConstraint("char_length(sha256) = 64", name="ck_evidence_sha_length"),
        Index("ix_evidence_alert", "alert_id"),
    )

    id: Mapped[str] = _pk()
    alert_id: Mapped[str] = mapped_column(ForeignKey("alert.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    bucket: Mapped[str] = mapped_column(String(64), nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    bytes: Mapped[int | None] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    enhanced: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    enhancement_params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    alert: Mapped[Alert] = relationship(back_populates="items")


class LedgerAnchorBatch(Base):
    __tablename__ = "ledger_anchor_batch"
    __table_args__ = (
        CheckConstraint("char_length(merkle_root) = 64", name="ck_batch_root_length"),
        CheckConstraint("leaf_count > 0", name="ck_batch_has_leaves"),
    )

    id: Mapped[str] = _pk()
    merkle_root: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    leaf_count: Mapped[int] = mapped_column(Integer, nullable=False)
    backend: Mapped[str] = mapped_column(String(16), nullable=False)
    tx_id: Mapped[str | None] = mapped_column(Text)
    block_number: Mapped[int | None] = mapped_column(Integer)
    anchored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    created_at: Mapped[datetime] = _now()


class LedgerAnchorEntry(Base):
    __tablename__ = "ledger_anchor_entry"
    __table_args__ = (
        UniqueConstraint("alert_id", name="uq_anchor_entry_alert"),
        Index("ix_anchor_entry_batch", "batch_id"),
    )

    id: Mapped[str] = _pk()
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("ledger_anchor_batch.id", ondelete="CASCADE")
    )
    alert_id: Mapped[str] = mapped_column(ForeignKey("alert.id", ondelete="CASCADE"))
    leaf_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    leaf_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # The sibling path: [["L","<hash>"], ["R","<hash>"], ...]
    proof: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()


class User(Base):
    __tablename__ = "app_user"  # "user" is reserved in PostgreSQL

    id: Mapped[str] = _pk()
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="viewer", nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = _now()
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "role IN ('viewer','operator','investigator','admin')", name="ck_user_role"
        ),
    )


class WatchlistPerson(Base):
    """Opt-in, and only reachable when faces are enabled (P6)."""

    __tablename__ = "watchlist_person"

    id: Mapped[str] = _pk()
    ref_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(32), default="person_of_interest")
    added_by: Mapped[str | None] = mapped_column(
        ForeignKey("app_user.id", ondelete="SET NULL")
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()


class WatchlistVehicle(Base):
    """Note what is absent: there is no plate_text column.

    You cannot leak a field that does not exist. Lookup is by HMAC; the
    ciphertext exists only for the rare, audited, legitimate reveal (P6, §7.9).
    """

    __tablename__ = "watchlist_vehicle"
    __table_args__ = (
        CheckConstraint("char_length(plate_hmac) = 64", name="ck_plate_hmac_len"),
    )

    id: Mapped[str] = _pk()
    plate_hmac: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    plate_ciphertext: Mapped[bytes | None] = mapped_column(LargeBinary)
    region: Mapped[str] = mapped_column(String(8), default="IN")
    category: Mapped[str] = mapped_column(String(32), default="watch")
    added_by: Mapped[str | None] = mapped_column(
        ForeignKey("app_user.id", ondelete="SET NULL")
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = _now()


class AuditLog(Base):
    """Append-only. Every evidence view and every plate reveal lands here (§12)."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_time", text("ts_utc DESC")),)

    id: Mapped[str] = _pk()
    actor_id: Mapped[str | None] = mapped_column(
        ForeignKey("app_user.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_type: Mapped[str | None] = mapped_column(String(32))
    target_id: Mapped[str | None] = mapped_column(String(64))
    ts_utc: Mapped[datetime] = _now()
    ip: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class StreamHealth(Base):
    __tablename__ = "stream_health"
    __table_args__ = (
        Index("ix_stream_health_camera_time", "camera_id", text("ts_utc DESC")),
    )

    id: Mapped[str] = _pk()
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id", ondelete="CASCADE"))
    ts_utc: Mapped[datetime] = _now()
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    fps_in: Mapped[float | None] = mapped_column(Float)
    fps_analytics: Mapped[float | None] = mapped_column(Float)
    evqm_profile: Mapped[str | None] = mapped_column(String(16))
    reconnects: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)


class ConfigVersion(Base):
    """Which thresholds produced this alert? This table is the answer (§11)."""

    __tablename__ = "config_version"

    id: Mapped[str] = _pk()
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    profile: Mapped[str] = mapped_column(String(16), nullable=False)
    site_code: Mapped[str | None] = mapped_column(String(32))
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    first_seen_at: Mapped[datetime] = _now()
