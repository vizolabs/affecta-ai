# ADR-005: Intensity Architecture

```
Status: PENDING
Created: 2026-09-10
```

---

## Decision

Which architecture should be used for expression intensity estimation?

---

## Ground Truth Definition

Before selecting architecture, define target:

```
Human rating: 1 ────── 5
              │       │
            Subtle   Obvious

Normalization: I = 100 * (mean_rating - 1) / 4
Output: 0-100 scale
```

---

## Options Considered

| Option | Architecture | Input | Output |
|--------|--------------|-------|--------|
| A | 2-layer MLP | Features + AUs | Scalar 0-100 |
| B | CNN head on backbone | Aligned face | Scalar 0-100 |
| C | Ordinal regression | Features + AUs | Ordinal class |

---

## Evaluation Criteria

| Metric | Target |
|--------|--------|
| MAE | < 15 |
| Correlation with human | > 0.70 |
| Category accuracy | > 75% |

---

## Experiment ID

EXP-005

---

## Status

PENDING

---

## Hypothesis

MLP with feature+AU input will be sufficient and fast.

---

## Result

[To be filled after experiment]

---

## Decision Date

[To be filled]

---

## Rationale

[To be filled after experiment]
