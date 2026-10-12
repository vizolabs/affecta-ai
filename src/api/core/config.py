"""API settings (env-driven, local defaults)."""

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load .env if present (python-dotenv; silent when the file is absent so a
# bare git clone still runs on built-in defaults).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv is optional
    pass

log = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r (expected int); using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r (expected float); using %s", name, raw, default)
        return default


class Settings:
    app_name: str = "Human Facial Emotion Detection"
    app_version: str = "1.0.0"

    # SQLite by default; set AFFECTA_DB_URL to switch (e.g. postgresql://...).
    # The DB filename is data/app.db; an existing legacy data/affecta.db is
    # adopted automatically so nobody loses their sessions on upgrade.
    @staticmethod
    def _default_db_url() -> str:
        legacy = PROJECT_ROOT / "data" / "affecta.db"
        modern = PROJECT_ROOT / "data" / "app.db"
        if legacy.exists() and not modern.exists():
            return f"sqlite:///{legacy}"
        return f"sqlite:///{modern}"

    db_url: str = os.environ.get("AFFECTA_DB_URL") or _default_db_url()

    # Token signing secret (local dev: generated once, persisted)
    token_secret_file: Path = PROJECT_ROOT / "data" / ".token_secret"
    token_ttl_seconds: int = _env_int("AFFECTA_TOKEN_TTL", 60 * 60 * 24 * 7)

    # CORS for the Next.js dev server. AFFECTA_CORS_ORIGINS overrides as a
    # comma-separated list; empty value means same-origin only (no CORS).
    _cors_raw = os.environ.get("AFFECTA_CORS_ORIGINS")
    cors_origins: list = (
        [o.strip() for o in _cors_raw.split(",") if o.strip()]
        if _cors_raw is not None
        else [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # ML
    default_model: str = os.environ.get("AFFECTA_MODEL", "dan")
    # Warm-load the model at startup (set 0 to skip, e.g. in tests)
    warm_model: bool = os.environ.get("AFFECTA_WARM_MODEL", "1").lower() not in (
        "0", "false", "no",
    )

    # Inference / events (spec 02 + 20)
    xai_frame_interval: int = 10
    intensity_change_threshold: float = 20.0
    confidence_drop_threshold: float = 20.0
    stabilization_seconds: float = 3.0
    face_exit_seconds: float = 2.0
    # Single-person product: the tracker keeps exactly one face (largest /
    # IoU-continuous), pinned to track id 0 so the narrator never invents a
    # second subject.
    max_faces: int = 1
    # Emotion must hold this long before a transition event fires (kills
    # A->B->A frame flicker). Tests set AFFECTA_TRANSITION_DEBOUNCE_SECONDS=0.
    transition_debounce_seconds: float = _env_float(
        "AFFECTA_TRANSITION_DEBOUNCE_SECONDS", 1.0
    )

    def get_token_secret(self) -> bytes:
        if self.token_secret_file.exists():
            return self.token_secret_file.read_bytes()
        import secrets

        secret = secrets.token_hex(32).encode()
        self.token_secret_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write (tmp + rename) so a crash never leaves a truncated
        # secret that would silently invalidate every existing token.
        tmp = self.token_secret_file.with_suffix(".tmp")
        tmp.write_bytes(secret)
        os.replace(tmp, self.token_secret_file)
        try:
            self.token_secret_file.chmod(0o600)
        except OSError:
            pass
        return secret


settings = Settings()
