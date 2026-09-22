# 04 — Ranking Engine (Core Research Contribution)

```
Status: LOCKED (concept) / EXPERIMENTAL (formula)
Version: V1.0
```

---

## 1. Research Question

**Does adding LRP-derived evidence improve ranking quality over probability + AU evidence alone?**

### Sub-questions
1. Is evidence-scored ranking better than softmax-only ranking?
2. Does AU evidence improve ranking?
3. Does LRP regional evidence improve ranking?
4. Does temporal evidence improve ranking?

---

## 2. Mathematical Formulation

### Stage 1: Preliminary Ranking (Fast Path)

```
S_preliminary = w_p * P_e + w_a * A_e
```

Where:
- P_e = classifier probability for emotion e
- A_e = AU evidence score for emotion e
- w_p, w_a = weights (sum to 1)

### Stage 2: Final Ranking (Slow Path, LRP available)

```
S_final = w_p * P_e + w_a * A_e + w_r * R_e + w_t * T_e
```

Where:
- P_e = classifier probability
- A_e = AU evidence score
- R_e = LRP regional evidence
- T_e = temporal evidence
- w_p + w_a + w_r + w_t = 1

---

## 3. Feature Definitions

### P_e (Classifier Probability)
Raw softmax output from expression model.

```python
P_e = softmax(expression_model(face))[e]
```

### A_e (AU Evidence Score)

```python
def compute_au_evidence(au_activations, emotion):
    au_weights = AU_TO_EMOTION_WEIGHTS[emotion]
    A_e = sum(au_weights[au] * au_activations[au] 
              for au in au_weights)
    return A_e
```

### R_e (LRP Regional Evidence)

```python
def compute_regional_evidence(lrp_heatmap, landmarks, emotion):
    regions = extract_facial_regions(landmarks)
    region_relevance = {}
    for region_name, region_mask in regions.items():
        region_relevance[region_name] = np.sum(
            np.abs(lrp_heatmap * region_mask)
        )
    # Normalize
    total = sum(region_relevance.values())
    for k in region_relevance:
        region_relevance[k] /= total
    
    # Weight by region importance for this emotion
    R_e = sum(REGION_WEIGHTS[emotion][r] * v 
              for r, v in region_relevance.items())
    return R_e, region_relevance
```

### T_e (Temporal Evidence)

```python
def compute_temporal_evidence(history, emotion, alpha=0.3):
    if len(history) < 2:
        return 0
    # Exponential moving average of emotion probability
    ema = history[0][emotion]
    for h in history[1:]:
        ema = alpha * h[emotion] + (1 - alpha) * ema
    # Current vs historical
    T_e = history[-1][emotion] - ema
    return T_e
```

---

## 4. Experiment Design

### Models to Compare

| ID | Model | Formula | Description |
|----|-------|---------|-------------|
| R0 | Softmax Only | S = P_e | Baseline |
| R1 | + AU | S = P_e + A_e | + facial movement evidence |
| R2 | + LRP | S = P_e + A_e + R_e | + regional evidence |
| R3 | + Temporal | S = P_e + A_e + R_e + T_e | + time evidence |

### Metrics

| Metric | Description |
|--------|-------------|
| Ranking Accuracy | Top-1 correct emotion |
| Pairwise Accuracy | Correct pairwise ordering |
| NDCG@3 | Normalized Discounted Cumulative Gain |

### Decision Criterion

```
If R2 significantly outperforms R1:
    → LRP adds value for ranking
    → Use R2 or R3

If R1 ≈ R2:
    → LRP not needed for ranking
    → Keep LRP for visualization only
    → Use R1
```

---

## 5. Weight Learning Strategy

### Option 1: Grid Search
```python
for w_p in [0.2, 0.3, 0.4, 0.5]:
    for w_a in [0.1, 0.2, 0.3]:
        for w_r in [0.1, 0.2, 0.3]:
            w_t = 1 - w_p - w_a - w_r
            if w_t < 0: continue
            evaluate(w_p, w_a, w_r, w_t)
```

### Option 2: Learned Weights
```python
# Small MLP that learns to combine evidence
ranking_mlp = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
```

### Option 3: Pairwise Ranking Loss
```python
def pairwise_ranking_loss(scores, targets):
    # scores: [B, 7] (one per emotion)
    # targets: [B] (true emotion index)
    loss = 0
    for i in range(7):
        if i != targets:
            loss += max(0, scores[i] - scores[targets] + margin)
    return loss
```

---

## 6. Three Ranking Outputs

### Emotion Ranking
```
1. Happy       0.88
2. Neutral     0.06
3. Surprise    0.03
4. Sad         0.02
5. Angry       0.01
```

### Region Ranking
```
1. Mouth       41%
2. Cheeks      28%
3. Eyes        19%
4. Eyebrows     8%
5. Other        4%
```

### AU Evidence Ranking
```
1. AU12        Strong (0.92)
2. AU6         Strong (0.87)
3. AU7         Moderate (0.45)
4. AU4         Weak (0.12)
```

---

## 7. Implementation

```python
class RankingEngine:
    """Evidence-scored ranking engine for facial expression analysis."""
    
    AU_TO_EMOTION = {
        'happy': {'AU6': 0.5, 'AU12': 0.5},
        'sad': {'AU4': 0.4, 'AU1': 0.3, 'AU15': 0.3},
        'angry': {'AU4': 0.3, 'AU5': 0.2, 'AU7': 0.3, 'AU23': 0.2},
        'fear': {'AU1': 0.2, 'AU2': 0.2, 'AU4': 0.2, 'AU5': 0.2, 'AU20': 0.2},
        'surprise': {'AU1': 0.2, 'AU2': 0.2, 'AU5': 0.3, 'AU26': 0.3},
        'disgust': {'AU9': 0.4, 'AU10': 0.4, 'AU15': 0.2},
        'neutral': {}
    }
    
    def __init__(self, weights=None):
        self.weights = weights or {'w_p': 0.4, 'w_a': 0.25, 
                                    'w_r': 0.25, 'w_t': 0.1}
    
    def compute_au_evidence(self, au_activations, emotion):
        evidence = 0
        for au, weight in self.AU_TO_EMOTION.get(emotion, {}).items():
            au_idx = self.AU_LIST.index(au)
            evidence += weight * au_activations[au_idx]
        return evidence
    
    def preliminary_ranking(self, probs, au_activations):
        scores = {}
        for i, emotion in enumerate(self.EMOTIONS):
            A_e = self.compute_au_evidence(au_activations, emotion)
            scores[emotion] = (self.weights['w_p'] * probs[i] + 
                               self.weights['w_a'] * A_e)
        return dict(sorted(scores.items(), key=lambda x: -x[1]))
    
    def final_ranking(self, probs, au_activations, 
                      region_evidence, temporal_evidence):
        scores = {}
        for i, emotion in enumerate(self.EMOTIONS):
            A_e = self.compute_au_evidence(au_activations, emotion)
            R_e = region_evidence.get(emotion, 0)
            T_e = temporal_evidence.get(emotion, 0)
            scores[emotion] = (
                self.weights['w_p'] * probs[i] +
                self.weights['w_a'] * A_e +
                self.weights['w_r'] * R_e +
                self.weights['w_t'] * T_e
            )
        return dict(sorted(scores.items(), key=lambda x: -x[1]))
```

---

## 8. Evaluation Checklist

- [ ] R0 baseline established
- [ ] R1 (+ AU) tested
- [ ] R2 (+ LRP) tested
- [ ] R3 (+ Temporal) tested
- [ ] Statistical significance tested
- [ ] Weight optimization completed
- [ ] ADR-002 updated with decision
