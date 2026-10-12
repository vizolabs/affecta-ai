"""Pydantic request/response schemas (spec 12 §3 response envelope)."""

import re
from datetime import datetime
from typing import Annotated, Any, Generic, List, Literal, Optional, TypeVar

from email_validator import EmailNotValidError, validate_email
from pydantic import AfterValidator, BaseModel, Field

# Local-first app: no mail is ever sent, so accept internal domains that
# email-validator always rejects (it only whitelists ".test" in test mode).
_LOCAL_DOMAIN_SUFFIXES = (
    ".local",
    ".localhost",
    ".localdomain",
    ".internal",
    ".lan",
    ".home",
)
_LOCAL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+$")


def _is_local_domain(domain: str) -> bool:
    return domain == "localhost" or domain.endswith(_LOCAL_DOMAIN_SUFFIXES)


def _validate_email(value: str) -> str:
    value = value.strip()
    if "@" in value:
        domain = value.rsplit("@", 1)[-1].lower()
        if _is_local_domain(domain):
            if not _LOCAL_SHAPE.match(value) or (
                "." not in domain and domain != "localhost"
            ):
                raise ValueError("Not a valid email address")
            return value
    try:
        result = validate_email(value, check_deliverability=False, test_environment=True)
    except EmailNotValidError as exc:
        raise ValueError(str(exc)) from exc
    return result.normalized


EmailStr = Annotated[str, AfterValidator(_validate_email)]


class ApiResponse(BaseModel):
    status: str = "success"  # 'success' | 'error'
    data: Any = None
    meta: Optional[dict] = None
    error: Optional[str] = None


T = TypeVar("T")


class Ok(BaseModel, Generic[T]):
    """Typed success envelope for OpenAPI (matches the `ok()` helper)."""

    status: str = "success"
    data: Optional[T] = None
    meta: Optional[dict] = None
    error: Optional[str] = None


# ---- Auth ----
class RegisterIn(BaseModel):
    email: EmailStr
    # Spec 12: 8-char minimum; 128 max keeps PBKDF2 bounded.
    password: str = Field(min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    # Max length is the PBKDF2 DoS guard — a huge password must be
    # rejected by validation before any hashing runs.
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserOut


# ---- Sessions ----
class SessionOut(BaseModel):
    id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    total_frames: int
    total_events: int
    avg_confidence: Optional[float] = None
    avg_intensity: Optional[float] = None
    metadata: Optional[dict] = None

    model_config = {"from_attributes": True}


class EventOut(BaseModel):
    id: str
    session_id: Optional[str] = None
    face_id: int
    timestamp: datetime
    event_type: str
    from_emotion: Optional[str] = None
    to_emotion: Optional[str] = None
    confidence: Optional[float] = None
    intensity: Optional[float] = None
    certainty: Optional[str] = None
    metadata: Optional[dict] = None

    model_config = {"from_attributes": True}


# ---- Settings ----
class SettingsOut(BaseModel):
    theme: str = "dark"
    narrator_enabled: bool = True
    narrator_speed: float = 1.0
    narrator_voice: str = "default"
    narrator_detail: str = "normal"
    ema_alpha: float = 0.35
    xai_frame_interval: int = 10
    analysis_fps: int = 8
    emotion_hold_seconds: float = 0.0
    analysis_mode: str = "live"
    analysis_interval_seconds: int = 120
    motion_trigger_enabled: bool = True

    model_config = {"from_attributes": True}


class SettingsUpdate(BaseModel):
    theme: Optional[Literal["dark", "light"]] = None
    narrator_enabled: Optional[bool] = None
    # Slider range in the UI: 0.5 – 2.0.
    narrator_speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    narrator_voice: Optional[str] = Field(default=None, max_length=256)
    narrator_detail: Optional[Literal["calm", "normal", "detailed"]] = None
    # Clamped to (0, 1]: alpha > 1 makes EMA probabilities invalid
    # (overshoot/negative weights) — never accept it from the wire.
    ema_alpha: Optional[float] = Field(default=None, ge=0.05, le=1.0)
    xai_frame_interval: Optional[int] = Field(default=None, ge=1, le=120)
    # Capture pacing: frames analysed per second (1-15).
    analysis_fps: Optional[int] = Field(default=None, ge=1, le=15)
    # Display dwell in seconds (0 = off).
    emotion_hold_seconds: Optional[float] = Field(default=None, ge=0.0, le=600.0)
    # "live" (continuous) or "periodic" (one analysis every N seconds).
    analysis_mode: Optional[Literal["live", "periodic"]] = None
    analysis_interval_seconds: Optional[int] = Field(default=None, ge=30, le=600)
    motion_trigger_enabled: Optional[bool] = None


# ---- Analysis payloads (WS + REST) ----
class FaceAnalysis(BaseModel):
    face_id: int
    box: List[int]
    emotion: str
    confidence: float
    probabilities: dict
    ranked_emotions: List[str]
    evidence_scores: dict
    certainty: str
    margin: float
    intensity: float
    intensity_category: str
    action_units: Optional[dict] = None
    au_model_status: str = "untrained"
    regions: Optional[dict] = None


class AnalysisOut(BaseModel):
    session_id: Optional[str] = None
    frame_id: Optional[int] = None
    faces: List[FaceAnalysis] = []
    events: List[dict] = []
    narration: List[dict] = []
    xai: Optional[dict] = None
    fps: Optional[float] = None
    model: Optional[str] = None
    timeline: Optional[dict] = None
    timestamp: Optional[float] = None


# ---- Analytics ----
class SummaryOut(BaseModel):
    session_id: Optional[str] = None
    duration_seconds: Optional[int] = None
    total_frames: int = 0
    total_events: int = 0
    avg_confidence: Optional[float] = None
    avg_intensity: Optional[float] = None
    model: Optional[str] = None
    expression_distribution: dict = {}
    event_type_counts: dict = {}
    transition_count: int = 0
    transition_matrix: dict = {}
    uncertainty_events: int = 0
    per_person: dict = {}


class TimelineOut(BaseModel):
    session_id: Optional[str] = None
    timeline: List[dict] = []
    events: List[EventOut] = []


class LoggedOutOut(BaseModel):
    logged_out: bool


class DeletedOut(BaseModel):
    deleted: str


class AccountDeletedOut(BaseModel):
    deleted: bool


def ok(data: Any, meta: Optional[dict] = None) -> dict:
    return {"status": "success", "data": data, "meta": meta, "error": None}


def fail(error: str, data: Any = None) -> dict:
    return {"status": "error", "data": data, "meta": None, "error": error}


def session_out(row) -> SessionOut:
    return SessionOut(
        id=row.id,
        start_time=row.start_time,
        end_time=row.end_time,
        duration_seconds=row.duration_seconds,
        total_frames=row.total_frames,
        total_events=row.total_events,
        avg_confidence=row.avg_confidence,
        avg_intensity=row.avg_intensity,
        metadata=row.meta,
    )


def event_out(row) -> EventOut:
    return EventOut(
        id=row.id,
        session_id=row.session_id,
        face_id=row.face_id,
        timestamp=row.timestamp,
        event_type=row.event_type,
        from_emotion=row.from_emotion,
        to_emotion=row.to_emotion,
        confidence=row.confidence,
        intensity=row.intensity,
        certainty=row.certainty,
        metadata=row.metadata_,
    )
