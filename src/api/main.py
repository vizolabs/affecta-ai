"""Human Facial Emotion Detection — FastAPI application (spec 12).

Run: uvicorn src.api.main:app  (or `make serve`)
"""

from __future__ import annotations

import logging
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
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("hfed.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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
        app.state.service = AnalysisService(warm=False)
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
    return ok({
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "model": settings.default_model,
        "model_loaded": bool(getattr(service, "loaded", False)),
    })
