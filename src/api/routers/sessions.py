"""Session management endpoints (spec 12 §2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from src.api.core.database import get_db
from src.api.core.deps import get_current_user
from src.api.models import EventRow, SessionRow, User
from src.api.schemas import event_out, ok, session_out

router = APIRouter()


def _get_owned(session_id: str, user: User, db: DBSession) -> SessionRow:
    row = db.get(SessionRow, session_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


@router.get("/api/sessions", response_model=None)
def list_sessions(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    rows = (
        db.query(SessionRow)
        .filter_by(user_id=user.id)
        .order_by(SessionRow.start_time.desc())
        .all()
    )
    return ok([session_out(r).model_dump(mode="json") for r in rows])


@router.get("/api/sessions/{session_id}", response_model=None)
def get_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    row = _get_owned(session_id, user, db)
    return ok(session_out(row).model_dump(mode="json"))


@router.get("/api/sessions/{session_id}/events", response_model=None)
def get_session_events(
    session_id: str,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    row = _get_owned(session_id, user, db)
    events = (
        db.query(EventRow)
        .filter_by(session_id=row.id)
        .order_by(EventRow.timestamp.asc())
        .all()
    )
    return ok([event_out(e).model_dump(mode="json") for e in events])


@router.delete("/api/sessions/{session_id}", response_model=None)
def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    row = _get_owned(session_id, user, db)
    db.delete(row)
    db.commit()
    return ok({"deleted": session_id})
