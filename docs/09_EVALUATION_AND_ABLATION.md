# 09 — Evaluation & Ablation Study

```
Status: LOCKED
Version: V1.0
```

---

## 1. Metrics by Component

### Expression Classifier

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall correct predictions | ≥ 75% |
| Precision (macro) | Average precision across classes | ≥ 0.70 |
| Recall (macro) | Average recall across classes | ≥ 0.70 |
| F1-Score (macro) | Harmonic mean of precision/recall | ≥ 0.70 |
| Confusion Matrix | Per-class performance | — |

### AU Detector

| Metric | Description | Target |
|--------|-------------|--------|
| F1 per AU | Per-AU F1 score | ≥ 0.60 |
| Mean F1 | Average across AUs | ≥ 0.65 |
| Precision | Average precision | ≥ 0.70 |
| Recall | Average recall | ≥ 0.60 |

### Ranking Engine

| Metric | Description | Target |
|--------|-------------|--------|
| Top-1 Accuracy | Correct top prediction | ≥ 80% |
| Pairwise Accuracy | Correct pairwise ordering | ≥ 75% |
| NDCG@3 | Ranking quality | ≥ 0.85 |

### Intensity

| Metric | Description | Target |
|--------|-------------|--------|
| MAE | Mean absolute error (if continuous) | < 15 |
| Correlation | With human ratings | > 0.70 |
| Category Accuracy | Low/Medium/High correct | > 75% |

### Uncertainty

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy (CERTAIN) | When model is certain | > 85% |
| Abstention Rate | Percentage marked ambiguous | < 20% |
| ECE | Expected Calibration Error | < 0.10 |

### XAI

| Metric | Description | Target |
|--------|-------------|--------|
| Deletion AUC | Prediction drop when removing regions | Lower is better |
| Insertion AUC | Prediction rise when adding regions | Higher is better |
| Stability | Heatmap consistency under perturbation | > 0.80 |

---

## 2. Ablation Study

### Models to Compare

| ID | Components | Purpose |
|----|------------|---------|
| M1 | VGG16 baseline | Establish baseline |
| M2 | VGG16 + augmentation | Augmentation effect |
| M3 | EfficientNet-B2 (or ResNet-50) | Backbone effect |
| M4 | M3 + AU branch | AU evidence effect |
| M5 | M4 + Ranking | Ranking effect |
| M6 | M5 + LRP | LRP evidence effect |
| M7 | M6 + Intensity | Intensity effect |
| M8 | M7 + Temporal | Temporal effect |

### Research Questions

| Question | Compare | Expected |
|----------|---------|----------|
| Does augmentation help? | M1 vs M2 | M2 > M1 |
| Does modern backbone help? | M2 vs M3 | M3 > M2 |
| Does AU evidence help? | M3 vs M4 | M4 > M3 |
| Does ranking help? | M4 vs M5 | M5 > M4 |
| Does LRP help ranking? | M5 vs M6 | M6 > M5 |
| Does intensity help? | M6 vs M7 | M7 > M6 |
| Does temporal help? | M7 vs M8 | M8 > M7 |

---

## 3. Cross-Dataset Testing

### Protocol

```
Train on: FER2013 + RAF-DB
Test on: CK+ (unseen dataset)

Report:
- Within-dataset accuracy (test split)
- Cross-dataset accuracy (CK+)
- Generalization gap = within - cross
```

### Goal

Demonstrate that the model generalizes beyond training data.

---

## 4. XAI Faithfulness Experiment

### Deletion Test

```python
def deletion_test(model, image, lrp_heatmap, landmarks):
    """Measure prediction drop when removing top regions."""
    regions = aggregate_regions(lrp_heatmap, landmarks)
    sorted_regions = sorted(regions.items(), key=lambda x: -x[1])
    
    baseline_prob = model(image).max().item()
    results = [baseline_prob]
    
    masked_image = image.clone()
    for region_name, _ in sorted_regions:
        mask = get_region_mask(landmarks, region_name)
        masked_image[mask] = 0  # Remove region
        prob = model(masked_image).max().item()
        results.append(prob)
    
    return results  # AUC of this curve
```

### Insertion Test

```python
def insertion_test(model, image, lrp_heatmap, landmarks):
    """Measure prediction rise when adding regions."""
    regions = aggregate_regions(lrp_heatmap, landmarks)
    sorted_regions = sorted(regions.items(), key=lambda x: -x[1])
    
    blank_image = torch.zeros_like(image)
    results = [model(blank_image).max().item()]
    
    for region_name, _ in sorted_regions:
        mask = get_region_mask(landmarks, region_name)
        blank_image[mask] = image[mask]  # Add region
        prob = model(blank_image).max().item()
        results.append(prob)
    
    return results  # AUC of this curve
```

---

## 5. Failure Mode Testing

### Test Conditions

| Category | Conditions |
|----------|------------|
| Lighting | Good, low light, backlight |
| Pose | Frontal, slight rotation, strong rotation |
| Occlusion | Glasses, mask, hand, partial face |
| Face size | Near, medium, far |
| Motion | Still, slow, fast |

### Report Format

```
Condition: Low Light
Accuracy: 68% (vs 82% normal)
Degradation: -14%

Condition: Strong Rotation
Accuracy: 61% (vs 82% normal)
Degradation: -21%
```

---

## 6. Evaluation Scripts

```
evaluation/
├── eval_expression.py      # Classifier metrics
├── eval_au.py              # AU detector metrics
├── eval_ranking.py         # Ranking metrics
├── eval_intensity.py       # Intensity metrics
├── eval_uncertainty.py     # Uncertainty metrics
├── eval_xai.py             # XAI faithfulness
├── eval_cross_dataset.py   # Cross-dataset testing
├── eval_failure_modes.py   # Failure mode testing
└── eval_ablation.py        # Ablation study
```
