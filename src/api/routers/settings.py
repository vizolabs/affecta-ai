"""User settings endpoints (spec 12 §2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from src.api.core.database import get_db
from src.api.core.deps import get_current_user
from src.api.models import SettingsRow, User
from src.api.schemas import Ok, Responses, SettingsOut, SettingsUpdate, ok

_UNAUTHORIZED: Responses = {401: {"description": "Invalid or expired credentials"}}

router = APIRouter(tags=["Settings"])


def _get_or_create(user: User, db: DBSession) -> SettingsRow:
    row = db.query(SettingsRow).filter_by(user_id=user.id).first()
    if row is None:
        row = SettingsRow(user_id=user.id)
        db.add(row)
        db.commit()
    return row


@router.get("/api/settings", response_model=Ok[SettingsOut], responses=_UNAUTHORIZED)
def get_settings(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    row = _get_or_create(user, db)
    return ok(SettingsOut.model_validate(row))


@router.put("/api/settings", response_model=Ok[SettingsOut], responses=_UNAUTHORIZED)
def update_settings(
    payload: SettingsUpdate,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    row = _get_or_create(user, db)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(row, field, value)
    db.commit()
    return ok(SettingsOut.model_validate(row))
