# 05 — Action Unit Module

```
Status: LOCKED
Version: V1.0
```

---

## 1. What Are Action Units?

The Facial Action Coding System (FACS) by Ekman decomposes facial expressions into individual muscle movements called Action Units (AUs).

Each AU represents a specific facial muscle activation:
- **AU4**: Brow Lowerer (corrugator supercilii)
- **AU6**: Cheek Raiser (orbicularis oculi)
- **AU7**: Lid Tightener (orbicularis oculi)
- **AU9**: Nose Wrinkler (levator labii superioris)
- **AU10**: Upper Lip Raiser (levator labii superioris)
- **AU12**: Lip Corner Puller (zygomatic major)

---

## 2. AU-to-Emotion Evidence Mapping

**Important:** AUs provide *supporting evidence*, not deterministic classification.

| AU | Name | Emotion Evidence |
|----|------|------------------|
| AU4 | Brow Lowerer | Supports Angry, Sad |
| AU6 | Cheek Raiser | Supports Happy |
| AU7 | Lid Tightener | Supports Angry |
| AU9 | Nose Wrinkler | Supports Disgust |
| AU10 | Upper Lip Raiser | Supports Disgust |
| AU12 | Lip Corner Puller | Supports Happy |

### Evidence Strength

```
AU12 detected → Provides supporting evidence for Happy classification
AU6 detected  → Provides supporting evidence for Happy classification
AU4 detected  → Provides supporting evidence for Angry classification
```

**NOT:** AU12 → Happy (deterministic)

---

## 3. Model Architecture

```python
class AUDetector(nn.Module):
    def __init__(self, num_aus=6):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_aus)
    
    def forward(self, x):
        features = self.backbone(x)
        activations = torch.sigmoid(features)
        return activations
```

- Input: 224×224×3 aligned face
- Output: 6 sigmoid activations (0-1 per AU)
- Loss: Binary Cross-Entropy per AU

---

## 4. Training Data

| Dataset | Subjects | AUs | Type |
|---------|----------|-----|------|
| BP4D | 41 | 8 | Posed + Spontaneous |
| DISFA | 135 | 8 | Spontaneous |

### Label Format
```
Subject_001_Frame_0001:
  AU4: 0 (absent)
  AU6: 1 (present)
  AU7: 1 (present)
  AU9: 0 (absent)
  AU10: 0 (absent)
  AU12: 1 (present)
```

---

## 5. AU Evidence Computation

```python
def compute_au_evidence(au_activations, emotion):
    """Compute AU evidence score for a specific emotion."""
    au_weights = AU_TO_EMOTION_WEIGHTS[emotion]
    evidence = 0
    for au_name, weight in au_weights.items():
        au_idx = AU_LIST.index(au_name)
        evidence += weight * au_activations[au_idx]
    return evidence
```

---

## 6. Integration with Ranking Engine

```
AU activations → AU evidence scores → Ranking engine
```

AU evidence is one of four inputs to the ranking formula:
- P_e (classifier probability)
- **A_e (AU evidence)** ← this module
- R_e (LRP regional evidence)
- T_e (temporal evidence)

---

## 7. Evaluation Metrics

| Metric | Target |
|--------|--------|
| F1-score per AU | ≥ 0.6 |
| Mean F1 across AUs | ≥ 0.65 |
| Precision | ≥ 0.7 |
| Recall | ≥ 0.6 |

---

## 8. Future Expansion

| Phase | AU Count | AUs |
|-------|----------|-----|
| V1 | 6 | AU4, AU6, AU7, AU9, AU10, AU12 |
| V2 | 12 | + AU1, AU2, AU5, AU15, AU17, AU20 |
| V3 | 20 | + AU23, AU24, AU25, AU26, AU27, AU28, AU43 |
