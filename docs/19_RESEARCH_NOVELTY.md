# 19 — Research Novelty & Limitations

```
Status: LOCKED
Version: V1.0
```

---

## 1. Core Contribution

An integrated evidence-ranking framework that combines:

1. **Facial-expression probabilities** from CNN classifier
2. **Action Unit evidence** from multi-label AU detector
3. **Region-level LRP relevance** from explainability engine
4. **Temporal information** from expression history

With:
- Separate expression-intensity estimation
- Uncertainty estimation
- Faithfulness evaluation of explanations

---

## 2. What Makes This Novel

### Not Just Classification + Heatmap

Traditional FER:
```
Image → Classifier → "Happy 91%"
```

AFFECTA:
```
Image → Classifier → "Happy 91%"
       → AU evidence → AU6 strong, AU12 strong
       → LRP → Mouth 41%, Cheeks 28%
       → Ranking → Combined evidence score
       → Intensity → 82/100
       → Uncertainty → CERTAIN
       → Explanation → Rule-based natural language
```

### Evidence-Scored Ranking

Not softmax sorting. Ranking based on:
- Classifier probability
- AU evidence
- LRP regional evidence
- Temporal evidence

### Three-Level Ranking

1. **Emotion ranking**: Which emotion is strongest
2. **Region ranking**: Which facial regions matter most
3. **AU evidence ranking**: Which facial movements are present

### Confidence ≠ Intensity

- Confidence: How sure the model is
- Intensity: How strong the expression is
- These are independent measures

### Ambiguity Detection

Detects when evidence is insufficient for reliable classification.

---

## 3. Comparison to Existing Work

| Aspect | Traditional FER | AFFECTA |
|--------|-----------------|---------|
| Output | Single emotion label | Ranked emotions + evidence |
| Explainability | None or heatmap | LRP + regions + AUs |
| Intensity | Not separated | Separate estimator |
| Uncertainty | Not detected | Ambiguity detection |
| Temporal | Single frame | Smoothed + transitions |
| Multi-face | Limited | Full tracking |

---

## 4. Known Limitations

### Technical

| Limitation | Impact | Future Work |
|------------|--------|-------------|
| AU detector limited to 6 AUs | Incomplete facial movement analysis | Expand to 20+ AUs |
| Temporal model is EMA | Simple smoothing only | LSTM/Transformer |
| LRP on-demand only | Not real-time XAI | Optimize LRP |
| No micro-expression detection | Misses brief expressions | Future feature |
| No voice/physiological signals | Single modality | Multimodal fusion |

### Scientific

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Observable expressions only | Cannot infer internal state | Clear terminology |
| Dataset bias | May not generalize | Cross-dataset testing |
| Label noise | Training uncertainty | Multiple annotators |
| Subject variability | Individual differences | Personalization |

### Ethical

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Cannot detect deception | By design | Clear disclaimers |
| Cannot diagnose conditions | Out of scope | Not claimed |
| Cultural variation | Expressions vary | Diverse training data |

---

## 5. Ethical Boundaries

### What We Do NOT Claim

- ❌ Lie detection
- ❌ Personality assessment
- ❌ Mental health diagnosis
- ❌ Intelligence estimation
- ❌ Criminality prediction
- ❌ Mind reading

### What We DO Claim

- ✅ Observable facial-expression analysis
- ✅ Model evidence explanation
- ✅ Expression strength estimation
- ✅ Temporal expression tracking
- ✅ Multi-face analysis

---

## 6. Future Work (V2.0+)

| Feature | Priority | Complexity |
|---------|----------|------------|
| Expand AU set (20+) | High | Medium |
| LSTM/Transformer temporal | High | High |
| Voice emotion integration | Medium | High |
| Micro-expression detection | Medium | High |
| Federated learning | Low | High |
| Personalization | Medium | Medium |
| Mobile deployment | Medium | Medium |

---

## 7. Academic Positioning

This work sits at the intersection of:

1. **Facial Expression Recognition** — Classification of expressions
2. **Explainable AI** — Understanding model decisions
3. **Action Unit Detection** — Facial muscle movement analysis
4. **Ranking Systems** — Ordering predictions by evidence
5. **Real-Time Systems** — Live video processing

### Research Gap Addressed

Most FER systems provide:
- Single prediction
- No explanation
- No evidence ranking
- No uncertainty

AFFECTA provides:
- Ranked predictions
- LRP-based explanation
- Multi-level evidence ranking
- Uncertainty detection
- Real-time multi-face analysis
