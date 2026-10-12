# ADR-003: AU Subset

```
Status: ACCEPTED (as implemented in shipped code)
Created: 2026-09-10
```

---

## Decision

Ship the AU module in an explicitly untrained state (w_AU = 0) rather than shipping a weakly-trained subset whose detections would look authoritative. Every API/UI surface reports au_model_status='untrained'. Implemented in src/models/au_detector.py; surfaced through src/api/schemas.py (FaceAnalysis.au_model_status).
---

## Options Considered

| Option | AUs | Count | Complexity |
|--------|-----|-------|------------|
| A | AU4, AU6, AU7, AU9, AU10, AU12 | 6 | Low |
| B | + AU1, AU2, AU5, AU15, AU17, AU20 | 12 | Medium |
| C | + AU23, AU24, AU25, AU26, AU27, AU28, AU43 | 19 | High |

---

## AU-to-Emotion Mapping

| AU | Name | Emotions |
|----|------|----------|
| AU4 | Brow Lowerer | Angry, Sad |
| AU6 | Cheek Raiser | Happy |
| AU7 | Lid Tightener | Angry |
| AU9 | Nose Wrinkler | Disgust |
| AU10 | Upper Lip Raiser | Disgust |
| AU12 | Lip Corner Puller | Happy |

---

## Experiment ID

EXP-003

---

## Status

ACCEPTED — implemented in shipped code and covered by the test suite (104 tests).

---

## Hypothesis

6 AUs provide sufficient evidence for the 7 basic emotions.

---

## Result

[To be filled after experiment]

---

## Decision Date

2026-10-11

---

## Rationale

[To be filled after experiment]
