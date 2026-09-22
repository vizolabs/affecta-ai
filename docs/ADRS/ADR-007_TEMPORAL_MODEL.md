# ADR-007: Temporal Model

```
Status: PENDING
Created: 2026-09-10
```

---

## Decision

Which temporal model should be used for expression smoothing and transition detection?

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

PENDING

---

## Hypothesis

EMA (Option A) will be sufficient for V1, with LSTM as future improvement.

---

## Result

[To be filled after experiment]

---

## Decision Date

[To be filled]

---

## Rationale

[To be filled after experiment]
