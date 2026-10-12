# Frontend — Human Facial Emotion Detection

Next.js 14 (App Router) + TypeScript + Tailwind CSS 3. Talks to the FastAPI
backend in `../src` over REST (`/api/*`) and a WebSocket
(`/ws/emotion-stream`).

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

Type-check (no JS test runner is configured — see `docs/17_TESTING_QA.md`):

```bash
npm run typecheck
```

## Structure

| Path | Purpose |
|------|---------|
| `src/app/(app)/` | Authenticated pages: dashboard, live, lab, analytics, settings |
| `src/app/page.tsx` | Login portal (shared `PortalFace` component) |
| `src/app/register/` | Registration portal |
| `src/app/login/` | Redirect stub → `/` (kept for bookmarks) |
| `src/components/` | UI (emotion, layout, common, auth, narrator, timeline) |
| `src/lib/` | API client, WS cadence helpers, session formatters, types |
| `src/stores/` | zustand stores (auth, theme, toast — localStorage keys `hfd-*`) |

## Conventions

- API base URL is `API_URL` in `src/lib/api.ts` (default `http://127.0.0.1:8765`).
- Every fetch goes through `apiFetch`, which attaches the bearer token,
  applies a 15s timeout, and redirects to `/` on 401.
- The live page authenticates the WebSocket with a first-message auth frame
  (never a `?token=` query param) and streams frames at the configured
  analysis rate (default 8 fps, range 1–15). Frames are never stored by the
  backend (see `docs/14_SECURITY_PRIVACY.md`).
