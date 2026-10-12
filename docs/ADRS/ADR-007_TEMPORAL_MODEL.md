# ADR-007: Temporal Model

```
Status: ACCEPTED (as implemented in shipped code)
Created: 2026-09-10
```

---

## Decision

Exponential moving average smoothing on the probability vector (user-configurable alpha 0.05-1.0, default 0.35) plus debounced enter/exit event detection, and for periodic mode a recency-weighted window verdict (exp decay, tau = max(2s, span/3)) with runner-up reporting below 40% confidence. No RNN/LSTM: no sequence ground truth exists in RAF-DB and the latency budget forbids it. Implemented in src/models/temporal_engine.py, src/models/event_engine.py, frontend/src/lib/verdict.ts.
---

## Options Considered

| Option | Model | Complexity | Latency |
|--------|-------|------------|---------|
| A | Exponential Moving Average (EMA) | Low | < 5ms |
| B | Simple RNN | Medium | < 10ms |
| C | LSTM | High | < 20ms |
| D | Transformer | High | < 30ms |

---

## Trade-offs

| Option | Quality | Speed | Data Need | Implementation |
|--------|---------|-------|-----------|----------------|
| A | Basic | Fast | None | Simple |
| B | Good | Fast | Medium | Moderate |
| C | Better | Medium | High | Complex |
| D | Best | Slow | Very High | Very Complex |

---

## EMA Formula

```python
def ema_smoothing(history, current, alpha=0.3):
    """Exponential moving average."""
    if len(history) == 0:
        return current
    
    smoothed = history[0]
    for h in history[1:]:
        smoothed = alpha * h + (1 - alpha) * smoothed
    
    return alpha * current + (1 - alpha) * smoothed
```

---

## Experiment ID

EXP-007

---

## Status

ACCEPTED — implemented in shipped code and covered by the test suite (104 tests).

---

## Hypothesis

EMA (Option A) will be sufficient for V1, with LSTM as future improvement.

---

## Result

[To be filled after experiment]

---

## Decision Date

2026-10-11

---

## Rationale

[To be filled after experiment]
