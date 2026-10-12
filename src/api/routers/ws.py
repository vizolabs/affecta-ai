"""WebSocket emotion stream (spec 12 §2).

Protocol
  Auth: ?token=<jwt-like> query param, or first text message = raw token
        (or JSON {"action":"auth","token":"..."}). Failure closes with 4001.
  Client -> server:
    binary message  : JPEG frame
    text  message   : JSON control {"action":"xai","face_id":0} | {"action":"ping"}
  Server -> client:
    {"type":"session","session_id":...}
    {"type":"analysis", ...faces/events/narration/xai/fps/timeline...}
    {"type":"xai", ...}   (on demand)
    {"type":"error","error":...}
    {"type":"pong"}
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, WebSocket
from sqlalchemy.exc import SQLAlchemyError
from starlette.websockets import WebSocketDisconnect

from src.api.core.config import settings
from src.api.core.database import SessionLocal
from src.api.core.security import verify_token
from src.api.models import EventRow, SessionRow, SettingsRow, XaiSnapshot
from src.api.services.inference import SessionAnalysis

router = APIRouter()

MAX_WS_CONNECTIONS_PER_USER = 30


def _load_user_prefs(user_id: str) -> dict:
    db = SessionLocal()
    try:
        row = db.query(SettingsRow).filter_by(user_id=user_id).first()
        if row is None:
            return {
                "narrator_enabled": True,
                "xai_interval": settings.xai_frame_interval,
                "narrator_detail": "normal",
                "ema_alpha": 0.35,
                "analysis_fps": 8,
                "emotion_hold_seconds": 0.0,
                "analysis_mode": "live",
                "analysis_interval_seconds": 120,
                "motion_trigger_enabled": True,
            }
        return {
            "narrator_enabled": row.narrator_enabled,
            "xai_interval": row.xai_frame_interval,
            "narrator_detail": getattr(row, "narrator_detail", "normal"),
            "ema_alpha": getattr(row, "ema_alpha", 0.35),
            "analysis_fps": getattr(row, "analysis_fps", 8),
            "emotion_hold_seconds": getattr(row, "emotion_hold_seconds", 0.0),
            "analysis_mode": getattr(row, "analysis_mode", "live"),
            "analysis_interval_seconds": getattr(row, "analysis_interval_seconds", 120),
            "motion_trigger_enabled": getattr(row, "motion_trigger_enabled", True),
        }
    finally:
        db.close()


def _persist_events(db, session_id: str, events) -> None:
    if not events:
        return
    rows = [
        EventRow(
            session_id=session_id,
            face_id=ev.face_id,
            timestamp=datetime.fromtimestamp(ev.timestamp, timezone.utc),
            event_type=ev.event_type,
            from_emotion=ev.from_emotion,
            to_emotion=ev.to_emotion,
            confidence=ev.confidence,
            intensity=ev.intensity,
            certainty=ev.certainty,
            metadata_=ev.metadata or None,
        )
        for ev in events
    ]
    db.add_all(rows)
    db.commit()


def _flush_timeline(db, session_row: SessionRow, analysis: SessionAnalysis, model: str) -> None:
    meta = dict(session_row.meta or {})
    meta["timeline"] = analysis.timeline
    meta["model"] = model
    session_row.meta = meta
    db.commit()


async def _read_token(websocket: WebSocket) -> Optional[str]:
    token = websocket.query_params.get("token")
    if token:
        return token
    try:
        first = await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        return None
    text = first.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if payload.get("action") == "auth":
            return payload.get("token")
        return payload.get("token")
    return first or None


@router.websocket("/ws/emotion-stream")
async def emotion_stream(websocket: WebSocket):
    await websocket.accept()

    token = await _read_token(websocket)
    user_id = verify_token(token) if token else None
    if user_id is None:
        await websocket.close(code=4001)
        return

    counts = websocket.app.state.ws_counts
    if counts.get(user_id, 0) >= MAX_WS_CONNECTIONS_PER_USER:
        await websocket.close(code=1013)
        return
    counts[user_id] = counts.get(user_id, 0) + 1

    prefs = _load_user_prefs(user_id)
    qp = websocket.query_params
    try:
        xai_interval = int(qp["xai_interval"]) if qp.get("xai_interval") else prefs["xai_interval"]
    except (TypeError, ValueError):
        xai_interval = prefs["xai_interval"]
    title = (qp.get("title") or "").strip()[:120] or None
    # Capture cadence is applied client-side; the values are recorded in the
    # session metadata so exports/analytics report the mode actually used.
    try:
        applied_fps = int(qp["fps"]) if qp.get("fps") else prefs["analysis_fps"]
    except (TypeError, ValueError):
        applied_fps = prefs["analysis_fps"]
    applied_fps = max(1, min(15, applied_fps))

    mode = (qp.get("mode") or prefs["analysis_mode"] or "live").strip().lower()
    if mode not in ("live", "periodic"):
        mode = "live"
    try:
        interval = int(qp["interval"]) if qp.get("interval") else prefs["analysis_interval_seconds"]
    except (TypeError, ValueError):
        interval = prefs["analysis_interval_seconds"]
    interval = max(30, min(600, interval))
    motion = qp.get("motion", "1") not in ("0", "false", "no")

    service = websocket.app.state.service
    analysis = SessionAnalysis(
        service,
        xai_interval=xai_interval,
        narrator_enabled=prefs["narrator_enabled"],
        narrator_detail=prefs["narrator_detail"],
        ema_alpha=prefs["ema_alpha"],
    )

    db = SessionLocal()
    session_row = SessionRow(
        user_id=user_id,
        start_time=datetime.now(timezone.utc),
        meta={
            **({"title": title} if title else {}),
            "applied_fps": applied_fps,
            "analysis_mode": mode,
            "applied_interval_seconds": interval if mode == "periodic" else None,
            "motion_trigger": motion if mode == "periodic" else False,
            "emotion_hold_seconds": prefs["emotion_hold_seconds"],
        },
    )
    db.add(session_row)
    db.commit()
    session_id = session_row.id
    t_start = time.time()
    total_events = 0

    await websocket.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                try:
                    # Heavy inference off the event loop so other requests
                    # (health, login, other streams) are not blocked.
                    result = await asyncio.to_thread(
                        analysis.process_jpeg, msg["bytes"]
                    )
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                    continue
                except RuntimeError as exc:
                    # e.g. model weights not installed on this machine.
                    await websocket.send_json({"type": "error", "error": str(exc)})
                    continue
                result["session_id"] = session_id

                frame_events = analysis.last_events
                _persist_events(db, session_id, frame_events)
                total_events += len(frame_events)

                if analysis.frame_id % 30 == 0:
                    _flush_timeline(db, session_row, analysis, service.model_name)

                await websocket.send_json({"type": "analysis", **result})
                continue

            if msg.get("text") is not None:
                await _handle_control(websocket, db, session_row, analysis, msg["text"])
                continue
    except WebSocketDisconnect:
        pass
    finally:
        try:
            session_row.end_time = datetime.now(timezone.utc)
            session_row.duration_seconds = int(time.time() - t_start)
            session_row.total_frames = analysis.frame_id
            session_row.total_events = total_events
            session_row.avg_confidence = analysis.avg_confidence
            session_row.avg_intensity = analysis.avg_intensity
            _flush_timeline(db, session_row, analysis, service.model_name)
        except SQLAlchemyError:
            # The session row can disappear underneath us (e.g. the account was
            # deleted from the danger zone while this stream was live). There is
            # nothing left to finalise, so drop the write quietly.
            db.rollback()
        finally:
            db.close()
            counts[user_id] = max(0, counts.get(user_id, 1) - 1)
            if counts[user_id] == 0:
                counts.pop(user_id, None)


async def _handle_control(websocket, db, session_row, analysis: SessionAnalysis, text: str) -> None:
    try:
        payload = json.loads(text)
        action = payload.get("action")
    except (json.JSONDecodeError, AttributeError):
        await websocket.send_json({"type": "error", "error": "invalid control message"})
        return

    if action == "ping":
        await websocket.send_json({"type": "pong"})
        return

    if action == "xai":
        try:
            face_id = int(payload.get("face_id", 0))
        except (TypeError, ValueError):
            await websocket.send_json(
                {"type": "error", "error": "invalid face_id"}
            )
            return
        result = await asyncio.to_thread(analysis.compute_xai, face_id)
        if result is None:
            await websocket.send_json(
                {"type": "error", "error": f"no tracked face {face_id}"}
            )
            return
        db.add(
            XaiSnapshot(
                session_id=session_row.id,
                face_id=face_id,
                emotion_label=result.get("emotion"),
                confidence=result.get("confidence"),
                region_relevance=result.get("regions"),
            )
        )
        db.commit()
        await websocket.send_json({"type": "xai", **result})
        return

    await websocket.send_json({"type": "error", "error": f"unknown action: {action}"})
