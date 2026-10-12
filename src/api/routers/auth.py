"""Authentication endpoints (spec 12 §2)."""

from __future__ import annotations

import time
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session as DBSession

from src.api.core.database import get_db
from src.api.core.deps import get_current_user
from src.api.core.security import (
    create_token,
    hash_password,
    revoke_token,
    verify_password,
)
from src.api.models import SettingsRow, User
from src.api.schemas import (
    AccountDeletedOut,
    LoggedOutOut,
    LoginIn,
    Ok,
    RegisterIn,
    TokenOut,
    UserOut,
    ok,
)

_UNAUTHORIZED = {401: {"description": "Invalid or expired credentials"}}

router = APIRouter()

# Minimal in-memory login rate limit (docs/14: 10 attempts / minute)
_attempts: Dict[str, List[float]] = {}


def _rate_limited(key: str, max_attempts: int = 10, window: float = 60.0) -> bool:
    now = time.time()
    hits = [t for t in _attempts.get(key, []) if now - t < window]
    _attempts[key] = hits
    return len(hits) >= max_attempts


def _record_failure(key: str) -> None:
    _attempts.setdefault(key, []).append(time.time())


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "local"


@router.post("/api/auth/register", response_model=Ok[TokenOut], responses={409: {"description": "Email already registered"}})
def register(payload: RegisterIn, db: DBSession = Depends(get_db)):
    existing = db.query(User).filter_by(email=payload.email.lower()).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    db.flush()
    db.add(SettingsRow(user_id=user.id))
    db.commit()
    token = create_token(user.id)
    return ok(TokenOut(token=token, user=UserOut.model_validate(user)).model_dump())


@router.post("/api/auth/login", response_model=Ok[TokenOut], responses={**_UNAUTHORIZED, 429: {"description": "Too many login attempts"}})
def login(payload: LoginIn, request: Request, db: DBSession = Depends(get_db)):
    key = _client_key(request)
    if _rate_limited(key):
        raise HTTPException(status_code=429, detail="Too many login attempts")
    user = db.query(User).filter_by(email=payload.email.lower()).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        _record_failure(key)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    _attempts.pop(key, None)
    from datetime import datetime, timezone

    user.last_login = datetime.now(timezone.utc)
    db.commit()
    token = create_token(user.id)
    return ok(TokenOut(token=token, user=UserOut.model_validate(user)).model_dump())


_bearer = HTTPBearer(auto_error=False)


@router.post("/api/auth/logout", response_model=Ok[LoggedOutOut])
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    user: User = Depends(get_current_user),
):
    if credentials is not None:
        revoke_token(credentials.credentials)
    return ok({"logged_out": True})


@router.get("/api/auth/me", response_model=Ok[UserOut], responses=_UNAUTHORIZED)
def me(user: User = Depends(get_current_user)):
    return ok(UserOut.model_validate(user))


@router.delete("/api/auth/me", response_model=Ok[AccountDeletedOut], responses=_UNAUTHORIZED)
def delete_me(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Danger zone: delete the account and all local sessions/settings."""
    if credentials is not None:
        revoke_token(credentials.credentials)
    db.delete(user)  # ORM cascades sessions, events, xai snapshots, settings
    db.commit()
    return ok({"deleted": True})
