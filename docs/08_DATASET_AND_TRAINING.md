# 08 — Dataset & Training Pipeline

```
Status: LOCKED (structure) / EXPERIMENTAL (specific access)
Version: V1.0
```

---

## 1. Dataset Compatibility Table

| Dataset | Expression | AU | Video | Size | Use | Access |
|---------|-----------|-----|-------|------|-----|--------|
| AffectNet | ✅ | — | — | 1M+ | Pre-train/Fine-tune | Verify |
| FER2013 | ✅ | — | — | 35K | Benchmark | Public |
| RAF-DB | ✅ | — | — | 30K | Generalization | Public |
| BP4D | — | ✅ | ✅ | 41 subj | AU training | Request |
| DISFA | — | ✅ | ✅ | 135 subj | AU training | Request |
| CK+ | ✅ | ✅ | ✅ | 123 subj | Cross-dataset | Request |
| ExpW | ✅ | — | — | 91K | Generalization | Public |

---

## 2. Label Harmonization

### 7-Class Scheme

```python
EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
```

### Mapping Rules

| Dataset | Labels | Mapping |
|---------|--------|---------|
| FER2013 | 7 classes | Direct mapping |
| RAF-DB | 7 classes | Direct mapping |
| AffectNet | 8 classes | Drop 'contempt' or merge with 'neutral' |
| CK+ | 7 classes + AU | Direct mapping |

### Multi-Label Handling

For datasets with multiple annotators:
```python
def resolve_label(annotations):
    """Majority vote for multi-annotator labels."""
    counts = Counter(annotations)
    return counts.most_common(1)[0][0]
```

---

## 3. Training Pipeline

### Phase 1: Pre-training (Optional)

```
Dataset: AffectNet (if accessible)
Task: Expression classification
Model: Backbone (EfficientNet/ResNet)
Epochs: 20
```

### Phase 2: Fine-tuning

```
Dataset: FER2013 + RAF-DB
Task: Expression classification
Model: Backbone + classifier head
Epochs: 50
Early stopping: patience=10
```

### Phase 3: AU Training

```
Dataset: BP4D + DISFA
Task: Multi-label AU detection
Model: ResNet-18 + sigmoid head
Epochs: 30
```

### Phase 4: Validation

```
Split: Held-out subjects (not seen during training)
Metrics: Accuracy, F1, confusion matrix
```

### Phase 5: Cross-Dataset Test

```
Train: FER2013 + RAF-DB
Test: CK+ (unseen)
Report: Generalization gap
```

---

## 4. Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.2
    ),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 5. Reproducibility

| Item | Specification |
|------|---------------|
| Random seed | 42 |
| Deterministic ops | torch.use_deterministic_algorithms(True) |
| Config file | configs/experiment_{id}.yaml |
| Package lock | requirements.txt with pinned versions |
| Dataset manifest | SHA256 checksums for all files |
| Model checkpoint | Hash stored in registry |
| Training logs | WandB or TensorBoard |

---

## 6. Dataset Access Checklist

- [ ] FER2013: Verify Kaggle access
- [ ] RAF-DB: Request access if needed
- [ ] AffectNet: Verify license and access
- [ ] BP4D: Request from authors
- [ ] DISFA: Request from authors
- [ ] CK+: Request from authors
- [ ] ExpW: Verify public access
