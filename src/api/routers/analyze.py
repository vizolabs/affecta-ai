"""Research Lab: photo + video analysis and model registry (docs/13)."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session as DBSession

from src.api.core.database import get_db, SessionLocal
from src.api.core.deps import get_current_user
from src.api.models import ModelRegistry, SettingsRow, User
from src.api.schemas import Responses, ok
from src.api.services.inference import SessionAnalysis
from src.api.services.video import VideoDecodeError, analyze_video

router = APIRouter(tags=["Research Lab"])

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 60 * 1024 * 1024

_UNAUTHORIZED: Responses = {401: {"description": "Invalid or expired credentials"}}
_NO_MODEL: Responses = {
    401: _UNAUTHORIZED[401],
    413: {"description": "Upload exceeds the size limit"},
    503: {"description": "Model failed to load (missing or corrupt weights — see README)"},
}


async def _read_bounded(file: UploadFile, limit: int) -> bytes:
    """Read the upload in chunks, aborting as soon as the limit is crossed.

    A naive `await file.read()` loads the whole body into RAM *before* any
    size check, so one multi-GB upload could exhaust memory.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {limit // (1024 * 1024)}MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _require_model(service) -> None:
    # One lazy retry: weights installed after startup recover without a
    # server restart (load_model is idempotent once loaded).
    if not getattr(service, "loaded", False):
        try:
            service.load_model()
        except Exception:
            pass
    if not getattr(service, "loaded", False):
        err = getattr(service, "load_error", None)
        if err:
            raise HTTPException(
                status_code=503,
                detail=f"Model failed to load: {err}",
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "Model weights are not installed on this machine — "
                "see README “Data & weights” for the download steps."
            ),
        )


@router.post("/api/analyze", response_model=None, responses=_NO_MODEL)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    xai: bool = Form(False),
    face_id: int = Form(0),
    user: User = Depends(get_current_user),
):
    data = await _read_bounded(file, MAX_UPLOAD_BYTES)
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty upload")

    service = request.app.state.service
    _require_model(service)
    session = SessionAnalysis(service)
    try:
        # Inference in the threadpool — a sync handler body would block the
        # event loop and stall every other request/stream.
        result = await asyncio.to_thread(
            session.process_jpeg, data, fallback_full_frame=True
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # MPS/CUDA OOM, cv2 errors, shape mismatches — surface a safe
        # message instead of a bare 500 (the WS path already does this).
        raise HTTPException(status_code=503, detail=f"Analysis failed: {exc}")

    if xai:
        target = face_id
        if target not in session.last_tracks and session.last_tracks:
            target = next(iter(session.last_tracks))
        payload = await asyncio.to_thread(session.compute_xai, target)
        if payload is not None:
            result["xai"] = payload

    return ok(result)


def _user_sample_fps(user_id: str, override: Optional[int] = None) -> int:
    """Sampling rate for video analysis: form override, else the user's fps setting."""
    if override:
        return max(1, min(int(override), 10))
    db = SessionLocal()
    try:
        row = db.query(SettingsRow).filter_by(user_id=user_id).first()
        fps = getattr(row, "analysis_fps", None) if row else None
    finally:
        db.close()
    # Cap the offline rate below the live setting: decoding + per-frame
    # inference over a whole clip should not saturate the CPU.
    return max(1, min(int(fps or 5), 10))


@router.post("/api/analyze/video", response_model=None, responses=_NO_MODEL)
async def analyze_video_upload(
    request: Request,
    file: UploadFile = File(...),
    sample_fps: int = Form(0),
    debounce_seconds: float = Form(-1.0),
    user: User = Depends(get_current_user),
):
    """Offline analysis of an uploaded clip (MP4/MOV/AVI/MKV/WEBM).

    `debounce_seconds` defaults to the server's live-stream value; pass >= 0 to
    override it (0 = report every sampled flip, useful for research).
    """
    data = await _read_bounded(file, MAX_VIDEO_BYTES)
    service = request.app.state.service
    _require_model(service)
    try:
        # Video decode + per-frame inference is the heaviest path in the app —
        # keep it off the event loop so the API stays responsive.
        payload = await asyncio.to_thread(
            analyze_video,
            service,
            data,
            filename=file.filename or "",
            content_type=file.content_type or "",
            sample_fps=_user_sample_fps(user.id, sample_fps or None),
            debounce_seconds=debounce_seconds if debounce_seconds >= 0 else None,
        )
    except VideoDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ok(payload)


@router.get("/api/models", response_model=None)
def list_models(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    rows = db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()
    return ok([
        {
            "model_name": r.model_name,
            "version": r.version,
            "model_type": r.model_type,
            "dataset_used": r.dataset_used,
            "metrics": r.metrics,
            "checkpoint_path": r.checkpoint_path,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ])
