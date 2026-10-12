"""Auth utilities: PBKDF2 password hashing + HMAC-signed bearer tokens.

Stdlib only (no bcrypt/jose dependency) — suitable for local single-machine use.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Optional

from src.api.core.config import settings

PBKDF2_ITERATIONS = 210_000

# In-memory token denylist (logout). Local single-process app.
REVOKED: dict = {}


def revoke_token(token: str) -> None:
    REVOKED[token] = time.time() + settings.token_ttl_seconds


def is_revoked(token: str) -> bool:
    now = time.time()
    for t, exp in list(REVOKED.items()):
        if exp < now:
            REVOKED.pop(t, None)
    return token in REVOKED


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS
    )
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), int(iterations)
        )
        return hmac.compare_digest(dk.hex(), digest)
    except (ValueError, AttributeError):
        return False


def _sign(payload: bytes) -> str:
    secret = settings.get_token_secret()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def create_token(user_id: str, ttl_seconds: Optional[int] = None) -> str:
    ttl = ttl_seconds if ttl_seconds is not None else settings.token_ttl_seconds
    payload = json.dumps(
        {"uid": user_id, "exp": int(time.time()) + ttl}, separators=(",", ":")
    ).encode()
    body = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"{body}.{_sign(body.encode())}"


def verify_token(token: str) -> Optional[str]:
    """Return user_id if token is valid and unexpired, else None."""
    try:
        body, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(_sign(body.encode()), sig):
        return None
    try:
        padded = body + "=" * (-len(body) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("exp", 0) < time.time():
        return None
    return data.get("uid")
