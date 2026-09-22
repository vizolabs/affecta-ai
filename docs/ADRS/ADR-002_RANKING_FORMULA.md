# ADR-002: Ranking Formula

```
Status: PENDING
Created: 2026-09-10
```

---

## Decision

Which ranking formula should be used for evidence-scored emotion ranking?

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

PENDING

---

## Hypothesis

R2 will significantly outperform R1, demonstrating that LRP evidence improves ranking.

---

## Result

[To be filled after experiment]

---

## Decision Date

[To be filled]

---

## Rationale

[To be filled after experiment]
