# ADR-005: Intensity Architecture

```
Status: ACCEPTED (as implemented in shipped code)
Created: 2026-09-10
```

---

## Decision

Use a distribution-extremity proxy (distance of the probability vector from uniform, scaled 0-100) instead of a separate intensity regressor — RAF-DB provides no intensity ground truth, and a proxy keeps the claim honest. The UI labels it as a proxy, never as a clinical measure. Implemented in src/models/intensity_estimator.py.
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

ACCEPTED — implemented in shipped code and covered by the test suite (109 tests).

---

## Hypothesis

MLP with feature+AU input will be sufficient and fast.

---

## Result

[To be filled after experiment]

---

## Decision Date

2026-10-11

---

## Rationale

[To be filled after experiment]
