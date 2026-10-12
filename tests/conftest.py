"""Shared pytest fixtures.

Env vars MUST be set before any src.api import: point the API at a
throwaway SQLite file and skip the production model warm-load.
"""

import os
import tempfile

_fd, _db_path = tempfile.mkstemp(prefix="affecta_test_", suffix=".db")
os.close(_fd)
os.environ["AFFECTA_DB_URL"] = f"sqlite:///{_db_path}"
os.environ["AFFECTA_WARM_MODEL"] = "0"
os.environ["AFFECTA_TOKEN_TTL"] = "3600"
# Integration tests script 2 frames back-to-back and expect an immediate
# transition; debounce logic is covered by unit tests with explicit values.
os.environ["AFFECTA_TRANSITION_DEBOUNCE_SECONDS"] = "0"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_auth_state():
    from src.api.core.security import REVOKED
    from src.api.routers import auth as auth_router

    REVOKED.clear()
    auth_router._attempts.clear()
    yield
    REVOKED.clear()
    auth_router._attempts.clear()
