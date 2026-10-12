"""SQLAlchemy ORM models — tables per docs/11_DATABASE_SCHEMA.md (SQLite-adapted).

UUIDs stored as CHAR(36) strings; JSONB -> JSON (portable across SQLite/Postgres).
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    sessions = relationship("SessionRow", back_populates="user", cascade="all, delete-orphan")
    settings_rel = relationship(
        "SettingsRow", back_populates="user", cascade="all, delete-orphan", uselist=False
    )


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    total_events: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float)
    avg_intensity: Mapped[Optional[float]] = mapped_column(Float)
    meta: Mapped[Optional[dict]] = mapped_column("metadata", JSON)

    user = relationship("User", back_populates="sessions")
    events = relationship(
        "EventRow", back_populates="session", cascade="all, delete-orphan"
    )
    xai_snapshots = relationship(
        "XaiSnapshot", back_populates="session", cascade="all, delete-orphan"
    )


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    face_id: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    from_emotion: Mapped[Optional[str]] = mapped_column(String(50))
    to_emotion: Mapped[Optional[str]] = mapped_column(String(50))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    intensity: Mapped[Optional[float]] = mapped_column(Float)
    certainty: Mapped[Optional[str]] = mapped_column(String(20))
    action_units: Mapped[Optional[dict]] = mapped_column(JSON)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON)

    session = relationship("SessionRow", back_populates="events")


class XaiSnapshot(Base):
    __tablename__ = "xai_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    face_id: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    emotion_label: Mapped[Optional[str]] = mapped_column(String(50))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    lrp_heatmap: Mapped[Optional[bytes]] = mapped_column()
    region_relevance: Mapped[Optional[dict]] = mapped_column(JSON)
    au_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)
    gradcam_heatmap: Mapped[Optional[bytes]] = mapped_column()
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON)

    session = relationship("SessionRow", back_populates="xai_snapshots")


class SettingsRow(Base):
    __tablename__ = "settings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    theme: Mapped[str] = mapped_column(String(20), default="dark")
    narrator_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    narrator_speed: Mapped[float] = mapped_column(Float, default=1.0)
    narrator_voice: Mapped[str] = mapped_column(String(50), default="default")
    narrator_detail: Mapped[str] = mapped_column(String(20), default="normal")
    ema_alpha: Mapped[float] = mapped_column(Float, default=0.35)
    xai_frame_interval: Mapped[int] = mapped_column(Integer, default=10)
    # Capture pacing (frames analysed per second upstream), 1-15.
    analysis_fps: Mapped[int] = mapped_column(Integer, default=8)
    # Display dwell: a detected emotion stays on screen this long before the
    # label may switch. 0 = off (label follows the live value immediately).
    emotion_hold_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    # "live" = continuous capture at analysis_fps; "periodic" = one analysis
    # every analysis_interval_seconds; "pulse" = scan window then pinned
    # hold while the next window accumulates (docs/13).
    analysis_mode: Mapped[str] = mapped_column(String(20), default="live")
    analysis_interval_seconds: Mapped[int] = mapped_column(Integer, default=120)
    # Pulse mode: seconds of frame accumulation per window (5-120) and
    # seconds each pooled verdict stays pinned on screen (5-120).
    pulse_scan_seconds: Mapped[int] = mapped_column(Integer, default=30)
    pulse_hold_seconds: Mapped[int] = mapped_column(Integer, default=30)
    # In periodic mode, significant movement forces an early re-analysis.
    motion_trigger_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user = relationship("User", back_populates="settings_rel")


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("model_name", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    dataset_used: Mapped[Optional[str]] = mapped_column(String(255))
    training_config: Mapped[Optional[dict]] = mapped_column(JSON)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    checkpoint_path: Mapped[Optional[str]] = mapped_column(String(500))
    checksum: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="trained")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
