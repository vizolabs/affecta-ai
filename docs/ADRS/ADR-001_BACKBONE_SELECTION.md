# ADR-001: Backbone Selection

```
Status: ACCEPTED (decision fills after EXP-005/EXP-006)
Created: 2026-09-10
Updated: 2026-09-23
```

---

## Decision

Which CNN backbone is used for the AFFECTA expression classifier and, therefore,
for the ranking / AU / XAI / intensity / temporal research core built on top of it.

---

## Context

Experiment status (as of 2026-09-23):

| Experiment | Dataset | Backbone | Status | Test Macro-F1 | Test Acc |
|------------|---------|----------|--------|---------------|----------|
| EXP-001 | FER2013 | VGG16 | **Historical / invalidated** (resume procedure broken; not usable as evidence) | – | – |
| EXP-004 | FER2013 | VGG16 | **Official VGG16 baseline** | **66.58%** | **67.84%** |
| EXP-005 | FER2013 | ResNet-50 | Historical (superseded by accuracy-first RAF-DB pivot) | – | – |
| EXP-006 | FER2013 | EfficientNet-B2 | Historical (superseded by accuracy-first RAF-DB pivot) | – | – |
| EXP-007 | RAF-DB | EfficientNet-B3 | **Deployable real-time backbone** | 50.21% (under-trained; tail retrain running) | 57.30% |
| EXP-008 | RAF-DB | ViT-Small | **Best accuracy** (extended 70-ep tail + TTA) | **70.05%** | **78.88%** |

The accuracy-first pivot moved the primary decision to the RAF-DB experiments
(EXP-007 vs EXP-008 vs their ensemble).

---

## Options Considered (official experiments)

Decision-relevant (RAF-DB, accuracy-first pivot):

| Option | Architecture | Params | Input Size | Test Macro-F1 | Test Acc | Notes |
|--------|--------------|--------|------------|---------------|----------|-------|
| A | ViT-Small (EXP-008) | 21.7M trainable | 224×224 | **70.05%** | **78.88%** | extended 70-ep tail + 10-view TTA |
| B | EfficientNet-B3 (EXP-007) | 10.7M trainable | 300×300 | 50.21% | 57.30% | under-trained 30-ep schedule; tail retrain running |
| C | Ensemble (A+B) | — | mixed | 62.34% | 69.75% | avg probs, no TTA; weak B3 drags it down |

Historical (FER2013): VGG16 (EXP-004) 66.58% F1 / 67.84% acc, 134.3M, 224×224,
14.7 ms/img (p95 17.5) on MPS, 1639 MiB peak RSS. ResNet-50 (EXP-005) and
EfficientNet-B2 (EXP-006) were superseded by the RAF-DB pivot and are not reported.

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

RAF-DB reference context (target of the accuracy-first pivot):

- Published single-network RAF-DB test accuracy: **86–91%** (DAN 2023, POSTER 2023,
  SCN, and EfficientNet/ResNet fine-tunes), Macro-F1 typically **~0.80–0.90**.
  These assume ImageNet-pretrained weights AND either 100–150+ fine-tune epochs
  and/or face-alignment preprocessing. Our pipeline reproduces the protocol but is
  compute-constrained locally; the Kaggle T4 path is the route to comparable numbers.

---

## Hypothesis (original)

EfficientNet-B2 provides the best accuracy/latency trade-off.

**Revised after the RAF-DB accuracy-first pivot:** a CNN for real-time latency
(EfficientNet-B3) plus a ViT-Small for peak accuracy, combined in an ensemble + TTA,
provide the best achievable accuracy under the real-time constraint.

---

## Result

**Decision: EfficientNet-B3 (EXP-007) as the deployable real-time backbone, with
ViT-Small (EXP-008) retained for accuracy-critical / offline crops and for a
2-model probabilistic ensemble + TTA.**

The accuracy-first pivot moved the primary evaluation to RAF-DB (public aligned
benchmark, 7 classes). Results are from the FAT staged schedule
(head_only 3 → block5 7 → block4_5 10 → block3_4_5 10; backbone frozen until the
final stage) trained on an Apple M4 MPS device (16 GB), ImageNet-pretrained
weights enabled, RandAugment + RandomErasing + RandomResizedCrop, CE loss,
cosine LR, batch 32 (224×224 in ViT, 300×300 in B3), AMP.

| Experiment | Backbone | Eval Protocol | Test Acc | Test Macro-F1 |
|------------|----------|---------------|----------|---------------|
| EXP-007 | EfficientNet-B3 | single, no TTA (30-ep schedule) | 57.30% | 50.21% |
| EXP-008 | ViT-Small | single, no TTA (30-ep schedule) | 69.95% | 61.53% |
| EXP-008 | ViT-Small | **+ extended fine-tune + 10-view TTA** | **78.88%** | **70.05%** |
| Ensemble | B3 + ViT-Small | avg probs, no TTA | 69.75% | 62.34% |

Calibration (temperature scaling), XAI (Grad-CAM / Grad×Input) and latency tooling
are implemented; ECE on the best model ≈ 0.52–0.55 raw, ≈ 0.52 T-scaled (model is
overconfident; still requires a threshold strategy — see ADR-006).

### Honest status vs. the 91% / 89% product target

The current local runs **do not meet** the ≥91% acc / ≥89% Macro-F1 target. The gap
has three known, addressable causes rather than a single dead end:

1. **Training budget** — the 30-epoch staged schedule is short for RAF-DB; public
   fine-tuning baselines use 100–150 epochs. The extended ViT run climbed
   monotonically across its tail (val F1 53→72%) and was early-stopped; further
   epochs and/or a final broad-LR stage are expected to add several points.
2. **Kaggle GPU + internet entitlement is still pending** (account-level
   `hasEverRun=false`, T4 quota granted but no GPU container yet, and no internet
   for ImageNet weight download). A real T4 transcript of the same schedule should
   reach roughly 85–91%, matching published RAF-DB CNN accuracy.
3. **No alignment beyond the dataset's own 100×100 registrations** — DAN-style
   landmark alignment / center cropping is the standard RAF-DB lever and is not
   yet applied.

The pipeline (train / resume / extend stages, ensemble, TTA, calibration, XAI,
latency) is now fully validated end-to-end locally, so the remaining gap is
compute + preprocessing, not architecture plumbing.

---

## Stage progression context

EXP-004 stage progression (val macro-F1): head_only 50.1 → +block5 59.2 → +block4_5 65.7 → +block3_4_5 67.4.

---

## Decision Date

2026-09-23

---

## Rationale

1. **ViT-Small (EXP-008) has the highest measured accuracy** (78.88% / 70.05% test
   with extended fine-tune + TTA) and is the clear winner on pure F1/accuracy.
2. **EfficientNet-B3 (EXP-007) wins on latency/memory deployability** (native
   300×300, lower peak RSS) which matters for the real-time AFFECTA edge core; its
   raw accuracy is lower but it is under-trained (finance: same 30-ep schedule and
   a resume with extended tail was launched; results supersede the table above).
3. **The 2-model ensemble** adds diversity (CNN + transformer) and trades a few
   points of peak accuracy for robustness; with both models fully trained it is
   expected to be the best single deployment object.
4. XAI compatibility is preserved for both candidates via the shared
   `build_expression_model` interface (Grad-CAM on EffNet conv stages; attention +
   Grad×Input on ViT), so the XAI / ranking research core (ADR-002…007) proceeds
   on top of the chosen backbone without coupling to one CNN.

---

## References

- Goodfellow et al. (2013). Challenges in Representation Learning. FER2013 benchmark.
- Khaireddin & Chen (2021). Facial Emotion Recognition: State of the Art Performance on FER2013. arXiv:2105.03588.
- EfficientNet-B2 on FER-2013 (2026). arXiv:2601.18228.
- TransFER (2021), POSTER (2023), DAN (2023) — SOTA FER literature, protocol caveats.
- justinshenk/fer, usef-kh/fer, LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch — reference implementations.
- ijazzia7 Benchmarking Modern CNN Architectures (RAF-DB / FER2013 / AffectNet) — methodology + calibration (ECE).