# ADR-004: LRP Frequency

```
Status: ACCEPTED (as implemented in shipped code)
Created: 2026-09-10
```

---

## Decision

LRP is offline-only (ml/xai, report generation, benchmarks). The real-time and photo pipelines serve Grad-CAM on demand (XAI inspector, analyze?xai=true) — LRP's per-pixel cost exceeds the latency budget for streamed frames. Implemented: ml/xai/* (offline), src/api/services/inference.py:gradcam (served).
---

## Options Considered

| Option | Description | Latency Impact |
|--------|-------------|----------------|
| A | Every frame | High (80ms added) |
| B | Every N frames | Medium (configurable) |
| C | Event-triggered only | Low (on transitions) |
| D | B + C combined | Low-Medium |

---

## Trade-offs

| Option | XAI Quality | Performance | Complexity |
|--------|-------------|-------------|------------|
| A | Best | Worst | Low |
| B | Good | Good | Low |
| C | Variable | Best | Medium |
| D | Good | Good | Medium |

---

## Experiment ID

EXP-004

---

## Status

ACCEPTED — implemented in shipped code and covered by the test suite (104 tests).

---

## Hypothesis

Option D (every N frames + event-triggered) provides the best balance.

---

## Result

[To be filled after experiment]

---

## Decision Date

2026-10-11

---

## Rationale

[To be filled after experiment]
