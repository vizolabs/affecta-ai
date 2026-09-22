# ADR-004: LRP Frequency

```
Status: PENDING
Created: 2026-09-10
```

---

## Decision

How often should LRP be computed in the real-time pipeline?

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

PENDING

---

## Hypothesis

Option D (every N frames + event-triggered) provides the best balance.

---

## Result

[To be filled after experiment]

---

## Decision Date

[To be filled]

---

## Rationale

[To be filled after experiment]
