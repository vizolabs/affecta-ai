# ADR-001: Backbone Selection

```
Status: ACCEPTED (decision fills after EXP-005/EXP-006)
Created: 2026-09-10
Updated: 2026-09-23 19:30
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
| EXP-007 | RAF-DB | EfficientNet-B3 | **Deployable real-time backbone** | 50.21% (tail retrain running; val F1 now ~59.4% and climbing) | 57.30% |
| EXP-008 | RAF-DB | ViT-Small | Extended 70-ep tail + TTA | 70.05% | 78.88% |
| EXP-009 | RAF-DB | DDAMFN++ (released ckpt) | **Architecture diversity for ensemble** (RetinaFace-realigned) | 73.40% (bacc) | **83.83%** |
| EXP-011 | RAF-DB | DAN (released ckpt) | **Accuracy champion** (reproduces published 89.70% exactly) | 82.75% (bacc) | **89.70%** |

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
| D | DAN (EXP-011) | released ckpt | 224×224 | 82.75% (bacc) | **89.70%** | released checkpoint reproduced exactly on our official split |
| E | DDAMFN++ (EXP-009) | released ckpt | 112×112 | 73.40% (bacc) | 83.83% | RetinaFace-realigned test; 82.40% on standard MTCNN-aligned test |

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

**Update 2026-09-23 (evening): accuracy champion established.** The released DAN
checkpoint (EXP-011) reproduces the published 89.70% exactly on our official test
split (balanced acc 82.75%), so DAN becomes the accuracy-critical production model
and the default webcam model. DDAMFN++ (EXP-009, 83.83% RetinaFace-aligned) provides
architecture diversity for the ensemble. Our in-house ViT-Small (78.88% TTA) and
EfficientNet-B3 remain the trainable/own-pipeline pair.

EXP-010 (`scripts/ensemble_production.py`) final numbers: uniform avg-probs
**ens_dan+ddamfn+vit = 89.99% acc / 82.90% bacc** (new best overall),
ens_dan+vit = 89.31% / **83.12% bacc** (best balanced),
ens_dan+ddamfn = 89.24% / 81.69% with ECE 0.018 (best-calibrated).
TTA **hurt** DAN (89.70 → 83.83) and DDAMFN++ (83.83 → 83.15) but helped ViT
(76.63 → 78.88), so the best ensemble is non-TTA. Temperature scaling on the
champion (val-fitted): T=0.257, ECE 0.113 → 0.0865.
Full leaderboard: `experiments/EXP-010-ENSEMBLE/production_ensemble_results.json`.

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

The gap is **substantially closed**: best overall is the EXP-010 3-model ensemble
at **89.99% accuracy** / 82.90% bacc; accuracy champion DAN (EXP-011) alone reaches
**89.70%** (paper-exact) with **82.75%** balanced accuracy, and ens_dan+vit reaches
**83.12%** bacc. The target was 89% macro-F1 — **not yet met** (82.90% best bacc <
89% target). Remaining path = longer fine-tune of the in-house models (now unblocked:
Kaggle T4 GPU confirmed working). Known contributors to the
remaining gap:

1. **Training budget** — the 30-epoch staged schedule is short for RAF-DB; public
   fine-tuning baselines use 100–150 epochs. The extended ViT run climbed
   monotonically across its tail (val F1 53→72%) and was early-stopped; further
   epochs and/or a final broad-LR stage are expected to add several points.
   EfficientNet-B3 tail retrain running (val F1 now ~59.4% and climbing).
2. **Kaggle GPU confirmed working** — phone verification completed 2026-09-23
   evening; GPU probe kernel ran to completion on a real **Tesla T4** with
   CUDA 13.0 and internet egress (evidence: `docs/evidence/kaggle_gpu_probe_2026-09-23.log`).
   GPU-kernel quota now unlocked; longer schedules can be moved to T4 if needed.
3. **No alignment beyond the dataset's own 100×100 registrations** — DAN-style
   landmark alignment / center cropping is the standard RAF-DB lever; the DDAMFN++
   RetinaFace realignment already showed +1.43 points (82.40 → 83.83%).

The pipeline (train / resume / extend stages, ensemble, TTA, calibration, XAI,
latency) is now fully validated end-to-end locally, so the remaining gap is
compute + preprocessing, not architecture plumbing.

---

## Stage progression context

EXP-004 stage progression (val macro-F1): head_only 50.1 → +block5 59.2 → +block4_5 65.7 → +block3_4_5 67.4.

---

## Decision Date

2026-09-23

Updated: 2026-09-23 19:30

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
- DAN — Dual Attention Network, arXiv:2304.03108; code+weights © yaoing/DAN (MIT), vendored at `ml/models/dan.py`.
- DDAMFN / DDAMFN++ — © SainingZhang/DDAMFN, vendored at `ml/models/ddam.py` + `ml/models/mixed_feature_net.py`.
- justinshenk/fer, usef-kh/fer, LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch — reference implementations.
- ijazzia7 Benchmarking Modern CNN Architectures (RAF-DB / FER2013 / AffectNet) — methodology + calibration (ECE).