# 17 — Testing & QA

```
Status: LOCKED
Version: V1.0
```

---

## 1. The Real Suite (as shipped)

One pytest suite: `pytest tests/` — **109 tests, all passing**. Run with
`make test`.

| File | Covers |
|------|--------|
| `tests/conftest.py` | Test app + SQLite fixtures, auth helpers |
| `tests/test_api.py` | REST: auth flow, register policy (8-char min, oversized rejected), settings clamps (ema_alpha ∈ [0.05, 1.0]), sessions CRUD + IDOR protection, analytics exports, health, 401/404/409/422 shapes |
| `tests/test_event_narrator.py` | Event engine debounce/enter-exit logic, narrator phrasing |
| `tests/test_models.py` | Ranking engine ordering, intensity proxy, uncertainty/margin logic |
| `tests/test_production_webcam.py` | Haar detector on synthetic frames, headless `--image` pipeline, device pick |
| `tests/test_tools.py` | LRP epsilon sanity, calibration ECE/T-scaling math, RAF-DB converter, evaluate |

A WebSocket integration test exercises the /ws/emotion-stream auth handshake and frame processing.

### Frontend

No JS test runner is configured (documented gap). UI behavior is verified by
`npm run build` + `tsc --noEmit` in the gate, plus manual/visual passes against
a running stack for the login portal, Live verdict mode, Lab and Analytics.

---

## 2. What the ML tests actually assert

- **Determinism**: same input tensor → same logits (`torch.allclose`).
- **Ranking consistency**: higher probability ranks higher; w_AU = 0 means AU
  activations never change ordering.
- **Calibration math**: ECE and temperature-scaling helpers reproduce known
  values (perfect calibration → ECE 0).
- **LRP epsilon**: relevance approximately conserves the output score.
- **Latency**: not asserted in CI (hardware-dependent); measured offline via
  `ml/benchmarks/latency.py`.

Not covered (honest gaps): lighting/pose/occlusion robustness suites, frontend
component tests, hosted-browser E2E. These were specified as future work in
early drafts of this document and are **not shipped**.

---

## 3. Manual QA Checklist (examiner walkthrough)

1. `make run` → register (min 8 chars) → login → dashboard.
2. Live: allow camera → verdict chip shows `VERDICT · EMOTION xx% · n=..`;
   settings cadence switch (live / verdict every N s) changes flush behavior.
3. Lab: analyze a sample photo → ranking + Grad-CAM + regions; API-down state
   shows a retry, not an infinite skeleton.
4. Analytics: export CSV/JSON/PDF — honesty footer present.
5. Settings: ema_alpha slider bounds; theme switch; narrator Test button.
6. Kill the API → header chip flips from "Model ready" to "API ok"/checking;
   photo analyze returns a 503 message pointing at the README.

---

## 4. Bug Tracking

### Labels

- `bug`: Something broken
- `feature`: New feature
- `documentation`: Docs update
- `performance`: Performance issue
- `security`: Security concern

### Priority

- `P0`: Critical, blocks development
- `P1`: High, should fix soon
- `P2`: Medium, normal priority
- `P3`: Low, nice to have
