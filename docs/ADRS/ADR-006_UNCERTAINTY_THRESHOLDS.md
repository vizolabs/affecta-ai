# ADR-006: Uncertainty Thresholds

```
Status: ACCEPTED (as implemented in shipped code)
Created: 2026-09-10
```

---

## Decision

Drive ambiguity events from the top-2 probability margin and calibrated confidence (ambiguity_enter / confidence_drop), debounced through the event engine so single noisy frames never flip state. Thresholds live in src/models/event_engine.py; uncertainty events are counted in the analytics summary (docs/15).
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

ACCEPTED — implemented in shipped code and covered by the test suite (104 tests).

---

## Hypothesis

Data-driven thresholds (Option B) will outperform initial guesses.

---

## Result

[To be filled after experiment]

---

## Decision Date

2026-10-11

---

## Rationale

[To be filled after experiment]
