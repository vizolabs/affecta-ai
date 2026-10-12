# 16 — Deployment Infrastructure

```
Status: LOCKED
Version: V1.0
```

---

## 1. Development Environment (as shipped)

Local-first: no containers, no cloud. Two processes on localhost.

```bash
make run          # API (uvicorn, 127.0.0.1:8765) + web (Next.js, 127.0.0.1:3000)
```

Or separately:

```bash
make serve        # uvicorn src.api.main:app --port 8765
make serve-web    # cd frontend && npm run dev
```

- Database: SQLite at `data/app.db` (created automatically; Postgres-compatible
  SQLAlchemy schema, but Postgres is **not** deployed or required).
- WebSocket emotion stream is served by the same uvicorn process
  (`/ws/analyze`), not a separate service.

---

## 2. Production Options

**Shipped:** single-machine local run (`make run`) — this is the intended
deployment. Privacy by design: inference is on-device, frames never leave the
machine, so a hosted multi-tenant deployment is deliberately out of scope.

Not shipped (would require work before use): Docker images, cloud deploys
(Vercel/Railway/AWS), TLS termination. If exposing beyond localhost, put a
reverse proxy with HTTPS in front of uvicorn and tighten CORS origins.

---

## 3. Environment Variables

### Backend (`src/api/core/config.py`)

```bash
AFFECTA_DB_URL=sqlite:///./data/app.db   # database URL (default SQLite)
AFFECTA_TOKEN_TTL=604800                 # bearer token TTL seconds (default 7 days)
AFFECTA_MODEL=dan                        # default production model name
AFFECTA_WARM_MODEL=1                     # 0 = start without loading weights
AFFECTA_TRANSITION_DEBOUNCE_SECONDS=1.0  # event debounce window
```

The token secret is generated on first run into `data/.token_secret`
(gitignored) — no env var needed.

### Frontend (`frontend/.env.local`)

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8765
```

---

## 4. Quality Gate (CI/CD not shipped)

No hosted CI is configured for this local-first project. The gate before any
commit is:

```bash
make test    # pytest tests/  (104 tests: REST, WebSocket, ML, narrator)
make lint    # flake8 + mypy (flake8 config in .flake8)
cd frontend && npm run build
```

---

## 5. Monitoring

### Health Check Endpoint (as implemented)

`GET /api/health` (src/api/main.py):

```json
{
  "status": "success",
  "data": {
    "status": "ok",
    "app": "Human Facial Emotion Detection",
    "version": "1.0.0",
    "model": "dan",
    "model_loaded": true
  }
}
```

`model_loaded: false` means the API is running degraded (weights not
installed) — analysis endpoints return 503 with a pointer to the README.

### Logging

`logging.basicConfig(level=INFO)` in `src/api/main.py`; model load failures
are logged with stack traces at startup. Structured request logging is not
shipped.

---

## 6. Model Deployment

Checkpoints are **not** redistributed in the repository. Expected paths
(`ml/models/production.py:DEFAULT_CKPTS`):

```
data/raw/dan_weights/RAF-DB.pth                                    # DAN (serving default)
experiments/EXP-009-DDAMFN-PP/checkpoints/rafdb_epoch20_acc0.9204_bacc0.8617.pth
experiments/EXP-008-VIT-SMALL/checkpoints/best.pt
experiments/EXP-007-EFFICIENTNET-B3/checkpoints/best.pt
```

See README “Data & weights” for acquisition steps. Without weights the app
still starts; the header chip reports the degraded state.
