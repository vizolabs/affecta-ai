"""Analytics + export endpoints (spec 12 §2, docs/15)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session as DBSession

from src.api.core.database import get_db
from src.api.core.deps import get_current_user
from src.api.models import EventRow, SessionRow, User
from src.api.schemas import event_out, ok
from src.api.services.report import (
    compute_summary,
    events_csv,
    export_json,
    report_pdf,
)

router = APIRouter()


def _get_owned(session_id: str, user: User, db: DBSession) -> SessionRow:
    row = db.get(SessionRow, session_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


def _load(row: SessionRow, db: DBSession):
    events = (
        db.query(EventRow)
        .filter_by(session_id=row.id)
        .order_by(EventRow.timestamp.asc())
        .all()
    )
    timeline = (row.meta or {}).get("timeline", [])
    return events, timeline


@router.get("/api/analytics/summary", response_model=None)
def get_summary(
    session_id: str,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    row = _get_owned(session_id, user, db)
    events, timeline = _load(row, db)
    return ok(compute_summary(row, events, timeline))


@router.get("/api/analytics/timeline", response_model=None)
def get_timeline(
    session_id: str,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    row = _get_owned(session_id, user, db)
    events, timeline = _load(row, db)
    return ok({
        "session_id": row.id,
        "timeline": timeline,
        "events": [event_out(e).model_dump(mode="json") for e in events],
    })


@router.get("/api/analytics/export/{session_id}", response_model=None)
def export_report(
    session_id: str,
    format: str = "pdf",
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    row = _get_owned(session_id, user, db)
    events, timeline = _load(row, db)
    fmt = format.lower()

    if fmt == "csv":
        return Response(
            content=events_csv(events),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="session_{row.id}.csv"'},
        )
    if fmt == "json":
        summary = compute_summary(row, events, timeline)
        return Response(
            content=export_json(row, summary, events, timeline),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="session_{row.id}.json"'},
        )
    if fmt == "pdf":
        summary = compute_summary(row, events, timeline)
        return Response(
            content=report_pdf(row, summary, events, timeline),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="session_{row.id}.pdf"'},
        )
    raise HTTPException(status_code=400, detail="format must be pdf, csv, or json")
