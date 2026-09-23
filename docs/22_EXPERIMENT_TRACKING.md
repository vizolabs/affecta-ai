# 22 — Experiment Tracking

```
Status: LOCKED
Version: V1.0
```

---

## 1. Purpose

Ensure reproducibility and track experimental progress.

---

## 2. Experiment Schema

```python
class Experiment:
    id: str                  # e.g., "EXP-001"
    name: str                # e.g., "Backbone Comparison"
    hypothesis: str          # What we're testing
    status: str              # "planned", "running", "completed"
    
    # Configuration
    dataset_version: str     # Dataset snapshot ID
    model_config: dict       # Hyperparameters
    seed: int                # Random seed
    
    # Results
    metrics: dict            # Evaluation results
    artifacts: list          # Checkpoints, plots, etc.
    
    # Metadata
    created_at: datetime
    completed_at: Optional[datetime]
    notes: str
```

---

## 3. Experiment IDs

Reserved/planned IDs (from the research roadmap):

| ID | Name | Status |
|----|------|--------|
| EXP-001 | Backbone Comparison | PLANNED |
| EXP-002 | Ranking Formula | PLANNED |
| EXP-003 | AU Subset | PLANNED |
| EXP-004 | LRP Frequency | PLANNED |
| EXP-005 | Intensity Architecture | PLANNED |
| EXP-006 | Uncertainty Thresholds | PLANNED |
| EXP-007 | Temporal Model | PLANNED |

Implemented runs (accuracy-first pivot, actual on-disk experiment IDs):

| ID | Dataset | Backbone | Protocol | Test Acc | Test Macro-F1 | Status |
|----|---------|----------|----------|----------|---------------|--------|
| EXP-004-VGG16-CLEAN | FER2013 | VGG16 | official baseline | 67.84% | 66.58% | COMPLETED (historical) |
| EXP-007-EFFICIENTNET-B3 | RAF-DB | EfficientNet-B3 | staged 30-ep schedule | 57.30% | 50.21% | COMPLETED |
| EXP-008-VIT-SMALL | RAF-DB | ViT-Small | staged 30-ep → extended tail + 10-view TTA | 78.88% | 70.05% | COMPLETED |
| EXP-009-DDAMFN-PP | RAF-DB | DDAMFN++ (released ckpt) | reproduced on our test (MTCNN / RetinaFace-realigned) | 82.40% / 83.83% | — (bacc 73.40% RF) | COMPLETED |
| EXP-010-ENSEMBLE | RAF-DB | DAN+DDAMFN++±ViT | avg-probs + TTA + T-scaling (val-fitted) | **ens_dan+ddamfn+vit 89.99%** (ens_dan+vit bacc 83.12%; T-scaling ECE 0.113→0.087) | bacc 82.90% | COMPLETED |
| EXP-011-DAN-REPRO | RAF-DB | DAN (released ckpt) | reproduced on our official 3068 test | **89.70%** | bacc 82.75% | COMPLETED (= published number) |

Deliverables per run: `experiments/<ID>/checkpoints/{best,last}.pt`, `test_metrics.json`,
`ensemble_*_test.json`. See ADR-001 for the analysis.

---

## 4. Tracking Structure

```
experiments/
├── EXP-001_backbone_comparison/
│   ├── config.yaml
│   ├── metrics.json
│   ├── checkpoints/
│   ├── plots/
│   └── notes.md
├── EXP-002_ranking_formula/
│   └── ...
└── ...
```

---

## 5. Config File Format

```yaml
# experiments/EXP-001/config.yaml
experiment:
  id: EXP-001
  name: Backbone Comparison
  hypothesis: "EfficientNet-B2 outperforms VGG16 and ResNet-50"

dataset:
  version: "v1.0"
  splits:
    train: ["FER2013", "RAF-DB"]
    val: ["FER2013-split"]
    test: ["CK+"]

models:
  - name: VGG16
    backbone: vgg16
    pretrained: true
    classifier: [4096, 4096, 7]
  
  - name: EfficientNet-B2
    backbone: efficientnet_b2
    pretrained: true
    classifier: [1408, 7]
  
  - name: ResNet-50
    backbone: resnet50
    pretrained: true
    classifier: [2048, 7]

training:
  optimizer: AdamW
  learning_rate: 0.0001
  batch_size: 32
  epochs: 50
  early_stopping: 10
  seed: 42

evaluation:
  metrics: ["accuracy", "macro_f1", "confusion_matrix"]
```

---

## 6. Metrics File Format

```json
{
    "experiment_id": "EXP-001",
    "completed_at": "2026-09-18T15:00:00Z",
    "results": {
        "VGG16": {
            "accuracy": 0.714,
            "macro_f1": 0.698,
            "latency_ms": 42
        },
        "EfficientNet-B2": {
            "accuracy": 0.742,
            "macro_f1": 0.731,
            "latency_ms": 51
        },
        "ResNet-50": {
            "accuracy": 0.736,
            "macro_f1": 0.725,
            "latency_ms": 44
        }
    },
    "decision": "EfficientNet-B2 selected",
    "rationale": "Best accuracy/F1 with acceptable latency"
}
```

---

## 7. Reproducibility Checklist

- [ ] Random seed fixed (42)
- [ ] Dataset version recorded
- [ ] Config file saved
- [ ] Package versions pinned (requirements.txt)
- [ ] Checkpoint hash recorded
- [ ] Training logs saved
- [ ] Evaluation scripts versioned
- [ ] Results documented

---

## 8. Integration with ADRs

Each experiment should update the corresponding ADR:

```markdown
# ADR-001: Backbone Selection

## Experiment ID
EXP-001

## Result
EfficientNet-B2 selected (accuracy: 0.742, F1: 0.731)

## Decision Date
2026-09-18

## Rationale
Best accuracy/F1 with acceptable latency (51ms)
```
