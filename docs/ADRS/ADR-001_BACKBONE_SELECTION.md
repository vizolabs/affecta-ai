# ADR-001: Backbone Selection

```
Status: PENDING (decision fills after EXP-005/EXP-006)
Created: 2026-09-10
Updated: 2026-09-22
```

---

## Decision

Which CNN backbone is used for the AFFECTA expression classifier and, therefore,
for the ranking / AU / XAI / intensity / temporal research core built on top of it.

---

## Context

Experiment status (as of 2026-09-22):

| Experiment | Backbone | Status | Test Macro-F1 | Test Acc |
|------------|----------|--------|---------------|----------|
| EXP-001 | VGG16 | **Historical / invalidated** (resume procedure broken; not usable as evidence) | – | – |
| EXP-004 | VGG16 | **Official VGG16 baseline** | **66.58%** | **67.84%** |
| EXP-005 | ResNet-50 | In progress | – | – |
| EXP-006 | EfficientNet-B2 | Pending | – | – |

This decision compares **only the three official experiments** (EXP-004 vs EXP-005 vs EXP-006).
EXP-001 is recorded as history only.

---

## Options Considered (official experiments)

| Option | Architecture | Parameters | Input Size | Test Macro-F1 | Test Acc | Latency (MPS, ms/img)* | Peak RSS (MiB)* |
|--------|--------------|------------|------------|---------------|----------|------------------------|-----------------|
| A | VGG16 (EXP-004) | 134.3M | 224×224 | **66.58%** | 67.84% | 14.7 (p95 17.5) | 1639 |
| B | ResNet-50 (EXP-005) | 23.5M | 224×224 | TBD | TBD | TBD | TBD |
| C | EfficientNet-B2 (EXP-006) | 7.7M | 260×260 | TBD | TBD | TBD | TBD |

*Latency = single-image forward pass on MPS (`ml/benchmarks/latency.py`, frozen best.pt).

Note: EfficientNet-B2 uses its native 260×260 input (architecture-dependent, weights
pretrained at that resolution); VGG16/ResNet-50 use 224×224.

---

## Evaluation Criteria (revised)

Primary criterion is Macro-F1 (FER2013 is imbalanced; accuracy alone hides minority-class
failure). Secondary criteria cover product fit for the real-time AFFECTA system.

| Order | Criterion | Weight | Description |
|-------|-----------|--------|-------------|
| 1 | **Macro-F1** | 30% | Test-set balanced macro-F1 (official single evaluation) |
| 2 | Accuracy | 15% | Test-set accuracy (official single evaluation) |
| 3 | Latency | 20% | Measured inference ms/image on MPS |
| 4 | Memory / params | 10% | Peak RSS + parameter count |
| 5 | Per-class robustness | 10% | Per-class F1 + confusion-matrix analysis (weak classes: fear/sad/angry) |
| 6 | XAI compatibility | 10% | LRP support (all three backbones expose staged blocks + get_features) |
| 7 | Training cost | 5% | Wall-clock epochs and checkpoint size |

---

## Verified reference context (protocol-audited only)

The following published numbers are offered as context, **not** as comparison targets.
Raw FER2013 numbers are not directly comparable across papers (different splits,
aligned/original images, class counts, and tuning budgets).

- Human accuracy on FER2013 ≈ **65 ± 5%** (Goodfellow et al., 2013).
- **VGG16** tuned single-network test accuracy: **73.28%** (Khaireddin & Chen, 2021, arXiv:2105.03588) — accuracy only, heavy hyperparameter tuning, no macro-F1 reported.
- **EfficientNet-B2** warm-up + fine-tune test accuracy: **68.78%**, ~9.2M params (arXiv:2601.18228, 2026) — accuracy only.
- ResNet-18 single-network: **73.70%** test accuracy (usef-kh / LetheSec repos).
- Reliability caveats (2025-2026 literature): several FER systems report high accuracy
  while macro-F1 is much lower (~0.50) on the same data; claims such as >90% on FER2013
  use non-comparable protocols and are excluded here.

---

## Hypothesis (original)

EfficientNet-B2 provides the best accuracy/latency trade-off.

---

## Result

[To be filled after EXP-005 and EXP-006 official evaluations.]

---

## Stage progression context

EXP-004 stage progression (val macro-F1): head_only 50.1 → +block5 59.2 → +block4_5 65.7 → +block3_4_5 67.4.

---

## Decision Date

[To be filled]

---

## Rationale

[To be filled after experiments]

---

## References

- Goodfellow et al. (2013). Challenges in Representation Learning. FER2013 benchmark.
- Khaireddin & Chen (2021). Facial Emotion Recognition: State of the Art Performance on FER2013. arXiv:2105.03588.
- EfficientNet-B2 on FER-2013 (2026). arXiv:2601.18228.
- TransFER (2021), POSTER (2023), DAN (2023) — SOTA FER literature, protocol caveats.
- justinshenk/fer, usef-kh/fer, LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch — reference implementations.
- ijazzia7 Benchmarking Modern CNN Architectures (RAF-DB / FER2013 / AffectNet) — methodology + calibration (ECE).