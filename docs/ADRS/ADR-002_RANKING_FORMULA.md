# ADR-002: Ranking Formula

```
Status: ACCEPTED (as implemented in shipped code)
Created: 2026-09-10
```

---

## Decision

Use calibrated softmax probabilities as the primary evidence score, ranked descending, with region-based Grad-CAM mass available as a reweighting signal. The AU branch carries weight w_AU = 0 until an AU model is trained — untrained signals are never allowed to influence rankings. Implemented in src/models/ranking_engine.py; verified by tests/test_models.py.
---

## Options Considered

| ID | Formula | Description |
|----|---------|-------------|
| R0 | S = P_e | Softmax only (baseline) |
| R1 | S = P_e + A_e | + AU evidence |
| R2 | S = P_e + A_e + R_e | + LRP evidence |
| R3 | S = P_e + A_e + R_e + T_e | + Temporal evidence |

---

## Research Question

Does adding LRP-derived evidence improve ranking quality over probability + AU evidence alone?

---

## Evaluation Criteria

| Metric | Target |
|--------|--------|
| Top-1 Accuracy | ≥ 80% |
| Pairwise Accuracy | ≥ 75% |
| NDCG@3 | ≥ 0.85 |

---

## Experiment ID

EXP-002

---

## Status

ACCEPTED — implemented in shipped code and covered by the test suite (104 tests).

---

## Hypothesis

R2 will significantly outperform R1, demonstrating that LRP evidence improves ranking.

---

## Result

[To be filled after experiment]

---

## Decision Date

2026-10-11

---

## Rationale

[To be filled after experiment]
