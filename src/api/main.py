"""Human Facial Emotion Detection — FastAPI application (spec 12).

Run: uvicorn src.api.main:app  (or `make serve`)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.core.config import settings
from src.api.core.database import init_db
from src.api.routers import analyze, analytics, auth, sessions, ws
from src.api.routers import settings as settings_router
from src.api.schemas import ok
from src.api.services.inference import AnalysisService

logging.basicConfig(
    level=os.environ.get("AFFECTA_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("hfed.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception:
        # A locked/corrupt/full SQLite file should not kill uvicorn with a
        # raw traceback — log a readable remediation and keep serving; DB
        # routes will 500 until the file is fixed, but /docs and /api/health
        # stay up for diagnosis.
        log.exception(
            "Database init failed — did you remove or corrupt data/app.db? "
            "Delete the file to recreate it, or set AFFECTA_DB_URL."
        )
    try:
        app.state.service = AnalysisService(warm=settings.warm_model)
    except Exception:
        # A fresh clone has no (gitignored) checkpoints — start degraded
        # instead of crashing. Analysis endpoints return 503 with a pointer
        # to the README until weights are installed.
        log.exception(
            "Model failed to load — starting without a model "
            "(analysis endpoints will return 503)"
        )
        try:
            app.state.service = AnalysisService(warm=False)
        except Exception:
            # Even the fallback can fail (broken OpenCV/cascade). Serve a
            # null service so _require_model 503s instead of the app dying.
            log.exception("Degraded service construction also failed")
            app.state.service = None
    app.state.ws_counts = {}
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(analytics.router)
app.include_router(settings_router.router)
app.include_router(analyze.router)
app.include_router(ws.router)


@app.get("/api/health")
def health():
    service = getattr(app.state, "service", None)
    load_error = getattr(service, "load_error", None)
    return ok({
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "model": settings.default_model,
        "model_loaded": bool(getattr(service, "loaded", False)),
        "model_error": load_error,
    })
