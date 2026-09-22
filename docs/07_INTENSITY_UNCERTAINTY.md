# 07 — Intensity & Uncertainty Estimation

```
Status: LOCKED (concept) / EXPERIMENTAL (architecture, thresholds)
Version: V1.0
```

---

## 1. Expression Intensity

### Definition
Strength of observed facial expression (0-100), **independent of confidence**.

### Key Distinction

```
Confidence ≠ Intensity

Example:
  Happy — 97% confidence — 34/100 intensity
  → Model is very sure, but expression is subtle

  Happy — 62% confidence — 89/100 intensity
  → Model is less sure, but expression is strong
```

### Ground Truth Definition

Before implementing the intensity model, we define the target:

```
Human rating scale: 1 ────── 5
                     │       │
                   Subtle   Obvious

Derivation:
  1-2 → Low intensity
  3   → Medium intensity
  4-5 → High intensity

Normalization:
  I = 100 * (mean_rating - 1) / 4

Output: 0-100 scale with semantic meaning
```

### Architecture (EXPERIMENTAL)

**Status: Define ground truth first, then select architecture.**

| Option | Architecture | Status |
|--------|--------------|--------|
| I1 | 2-layer MLP | EXPERIMENTAL |
| CNN head on backbone | EXPERIMENTAL |
| I3 | Ordinal regression | EXPERIMENTAL |

```python
class IntensityEstimator(nn.Module):
    def __init__(self, input_dim=512+6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1, scale to 0-100
        )
    
    def forward(self, features, au_activations):
        x = torch.cat([features, au_activations], dim=1)
        intensity = self.fc(x) * 100
        return intensity
```

### Intensity Categories

| Range | Category | Display |
|-------|----------|---------|
| 0-33 | Low | Weak expression |
| 34-66 | Medium | Moderate expression |
| 67-100 | High | Strong expression |

---

## 2. Uncertainty / Ambiguity Engine

### Purpose
Detect when the model cannot reliably classify the expression.

### States

| State | Condition | Display |
|-------|-----------|---------|
| CERTAIN | High confidence, clear margin | Normal display |
| AMBIGUOUS | Low margin between top-2 | ⚠ Warning |
| INSUFFICIENT | Low max probability | ⚠ "Analysis paused" |

### Algorithm

```python
def compute_uncertainty(probs, margin_threshold=0.15, 
                        prob_threshold=0.4):
    """
    Compute uncertainty state from emotion probabilities.
    
    Args:
        probs: [7] softmax probabilities
        margin_threshold: min gap for certainty
        prob_threshold: min probability for certainty
    
    Returns:
        state: 'certain' | 'ambiguous' | 'insufficient'
        margin: gap between top-2 emotions
    """
    sorted_probs = torch.sort(probs, descending=True)[0]
    top1 = sorted_probs[0]
    top2 = sorted_probs[1]
    
    margin = top1 - top2
    
    if top1 < prob_threshold:
        return 'insufficient', margin
    elif margin < margin_threshold:
        return 'ambiguous', margin
    else:
        return 'certain', margin
```

### Threshold Selection (EXPERIMENTAL)

**Status: Learn from validation data, don't hard-code.**

```python
# Threshold optimization
best_threshold = None
best_score = 0

for margin_thresh in [0.1, 0.15, 0.2, 0.25]:
    for prob_thresh in [0.3, 0.35, 0.4, 0.45]:
        states = [compute_uncertainty(p, margin_thresh, prob_thresh) 
                  for p in val_probs]
        
        # Evaluate: accuracy when certain vs. abstention rate
        certain_mask = [s == 'certain' for s in states]
        if sum(certain_mask) > 0:
            acc_certain = accuracy[val_targets[certain_mask], 
                                   val_preds[certain_mask]]
            abstention_rate = 1 - sum(certain_mask) / len(states)
            
            # Balance accuracy and abstention
            score = acc_certain - 0.5 * abstention_rate
            
            if score > best_score:
                best_score = score
                best_threshold = (margin_thresh, prob_thresh)
```

### Calibration

```
1. Train model
2. Run on validation set
3. Collect probability distributions
4. Compute Expected Calibration Error (ECE)
5. Apply temperature scaling if needed
6. Re-compute thresholds
```

---

## 3. Integration

### Output Structure

```python
{
    'emotion': 'happy',
    'confidence': 0.91,
    'ranking': {'happy': 0.91, 'neutral': 0.06, ...},
    'intensity': 82,
    'intensity_category': 'high',
    'certainty': 'certain',
    'margin': 0.85,
    'aus': {'AU6': 0.87, 'AU12': 0.92, ...},
    'regions': {'mouth': 0.41, 'cheeks': 0.28, ...}
}
```

---

## 4. Evaluation

### Intensity
| Metric | Target |
|--------|--------|
| MAE (if continuous) | < 15 |
| Correlation with human ratings | > 0.7 |
| Category accuracy | > 75% |

### Uncertainty
| Metric | Target |
|--------|--------|
| Accuracy when CERTAIN | > 85% |
| Abstention rate | < 20% |
| ECE | < 0.1 |
