# ADR-006: Uncertainty Thresholds

```
Status: PENDING
Created: 2026-09-10
```

---

## Decision

What thresholds should be used for the uncertainty/ambiguity engine?

---

## Options Considered

| Option | Margin Threshold | Prob Threshold | Source |
|--------|------------------|----------------|--------|
| A | 0.15 | 0.40 | Initial guess |
| B | Learned from validation | Learned | Data-driven |
| C | Temperature scaling | Calibrated | Calibration-based |

---

## Algorithm

```python
def compute_uncertainty(probs, margin_threshold, prob_threshold):
    sorted_probs = torch.sort(probs, descending=True)[0]
    margin = sorted_probs[0] - sorted_probs[1]
    
    if sorted_probs[0] < prob_threshold:
        return 'insufficient'
    elif margin < margin_threshold:
        return 'ambiguous'
    else:
        return 'certain'
```

---

## Evaluation Criteria

| Metric | Target |
|--------|--------|
| Accuracy when CERTAIN | > 85% |
| Abstention rate | < 20% |
| ECE | < 0.10 |

---

## Experiment ID

EXP-006

---

## Status

PENDING

---

## Hypothesis

Data-driven thresholds (Option B) will outperform initial guesses.

---

## Result

[To be filled after experiment]

---

## Decision Date

[To be filled]

---

## Rationale

[To be filled after experiment]
