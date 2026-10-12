# Frontend — Human Facial Emotion Detection

Next.js 15 (App Router) + TypeScript + Tailwind CSS 4. Talks to the FastAPI
backend in `../src` over REST (`/api/*`) and a WebSocket (`/ws/analyze`).

## Run it

```bash
# 1. start the backend (from the repo root)
make run                      # serves http://127.0.0.1:8765

# 2. start this frontend
npm install
npm run dev                   # http://localhost:3000
```

Production build:

```bash
npm run build && npm start
```

Type-check (no test runner is configured — see `docs/17_TESTING_QA.md`):

```bash
npx tsc --noEmit
```

## Structure

| Path | Purpose |
|------|---------|
| `src/app/(app)/` | Authenticated pages: dashboard, live, lab, analytics, settings |
| `src/app/(auth)/` | Login / register portals (shared `PortalFace` component) |
| `src/app/api/` | Next.js route proxies → backend `127.0.0.1:8765` |
| `src/components/` | UI (emotion, layout, common, auth) |
| `src/lib/` | API client, WS client, cadence/timer helpers, types |
| `src/stores/` | zustand stores (auth, theme — localStorage keys `hfd-*`) |

## Conventions

- API base URL is `API_BASE` in `src/lib/api.ts` (default `http://127.0.0.1:8765`).
- Every fetch goes through `apiFetch`, which attaches the bearer token and
  redirects to `/login` on 401.
- The live page streams frames to the backend at ~10 fps; frames are never
  stored by the backend (see `docs/14_SECURITY_PRIVACY.md`).
