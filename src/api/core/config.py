"""API settings (env-driven, local defaults)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings:
    app_name: str = "Human Facial Emotion Detection"
    app_version: str = "1.0.0"

    # SQLite by default; set AFFECTA_DB_URL to switch (e.g. postgresql://...)
    db_url: str = os.environ.get(
        "AFFECTA_DB_URL", f"sqlite:///{PROJECT_ROOT / 'data' / 'affecta.db'}"
    )

    # Token signing secret (local dev: generated once, persisted)
    token_secret_file: Path = PROJECT_ROOT / "data" / ".token_secret"
    token_ttl_seconds: int = int(os.environ.get("AFFECTA_TOKEN_TTL", str(60 * 60 * 24 * 7)))

    # CORS for the Next.js dev server
    cors_origins: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

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
    transition_debounce_seconds: float = float(
        os.environ.get("AFFECTA_TRANSITION_DEBOUNCE_SECONDS", "1.0")
    )

    def get_token_secret(self) -> bytes:
        if self.token_secret_file.exists():
            return self.token_secret_file.read_bytes()
        import secrets

        secret = secrets.token_hex(32).encode()
        self.token_secret_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_secret_file.write_bytes(secret)
        try:
            self.token_secret_file.chmod(0o600)
        except OSError:
            pass
        return secret


settings = Settings()
