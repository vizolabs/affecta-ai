# AFFECTA AI

**"A Ranking-Based Explainable AI Model for Human Facial Emotion Detection"**

A real-time facial expression analysis system with evidence-ranked explainability, Action Unit detection, LRP/Grad-CAM explanations, narrator-style commentary, and multi-face tracking.

## Headline results (RAF-DB official test split, n=3068)

| Rank | Model | Test Acc | Balanced Acc | Notes |
|------|-------|----------|--------------|-------|
| 1 | **ens_dan+ddamfn+vit** (EXP-010) | **89.99%** | 82.90% | uniform avg-probs 3-model ensemble |
| 2 | **DAN** (EXP-011) | **89.70%** | 82.75% | released ckpt, paper-exact reproduction |
| 3 | ens_dan+vit (EXP-010) | 89.31% | **83.12%** | best balanced accuracy |
| 4 | ens_dan+ddamfn (EXP-010) | 89.24% | 81.69% | ECE 0.018 (best-calibrated) |
| 5 | DDAMFN++ (EXP-009) | 83.83% | 73.40% | released ckpt, RetinaFace-realigned test |
| 6 | ViT-Small + 10-view TTA (EXP-008) | 78.88% | 73.23% | staged fine-tune (in-house) |
| 7 | EfficientNet-B3 (EXP-007) | 57.30% | 50.21 F1 | staged 30-ep schedule (in-house) |

- DAN exactly reproduces the published 89.70% (arXiv:2304.03108) on our official split.
- Temperature scaling on the champion ensemble: ECE 0.113 → 0.087 (T=0.257, val-fitted).
- TTA helps ViT (+2.25) but *hurts* DAN (89.70 → 83.83) — the 3-model ensemble without
  TTA is the best overall. Full leaderboard: `experiments/EXP-010-ENSEMBLE/production_ensemble_results.json`.

## Architecture

- **Frontend**: React/Next.js + TypeScript
- **Backend**: FastAPI (Python)
- **ML Models**: PyTorch (DAN / DDAMFN++ / ViT-Small / EfficientNet-B3)
- **Database**: PostgreSQL
- **Real-time**: WebSocket/WebRTC

## Emotion Classes

Canonical order (`src/core.config.EMOTIONS`, alphabetical — all production models re-map to this order internally):

| # | Emotion |
|---|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

## Project Structure

```
├── src/                    # Application code
│   ├── core/               # Core config (EMOTIONS, InferenceConfig)
│   ├── models/             # Classifiers, AU, ranking, intensity, temporal
│   ├── api/                # FastAPI backend
│   └── pipeline/           # Real-time pipeline + webcam module
├── ml/                     # Training, evaluation, XAI, benchmarks
│   ├── data/               # Datasets + augmentation (incl. picklable TTA)
│   ├── models/             # backbones, production zoo (DAN, DDAMFN++), ddam
│   ├── training/           # staged trainer, checkpointing, losses
│   ├── xai/                # LRP-epsilon, Grad-CAM, region mapping
│   └── benchmarks/         # calibration (ECE/T-scaling), latency, XAI probe
├── scripts/                # prepare_rafdb, ensemble_production, ensemble_eval
├── tests/                  # pytest suite (45 tests, all passing)
├── experiments/            # EXP-004…011 results + checkpoints
├── docs/                   # numbered specs 00–22 + ADRs
└── data/                   # raw + processed RAF-DB (gitignored)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt   # opencv pinned <5 (OpenCV 5 dropped Haar cascades)
```

### 2. Setup Dataset (RAF-DB)

Raw parquets live in `data/raw/rafdb/`. Prepare the train/val/test image splits:

```bash
python scripts/prepare_rafdb.py --raw data/raw/rafdb --out data/processed/rafdb
```

### 3. Live Webcam Emotion Detection

```bash
# Default: DAN (89.70% test acc), camera 0, EMA smoothing + HUD
python -m src.pipeline.webcam

# Options
python -m src.pipeline.webcam --model ddamfn      # DDAMFN++ (112px input)
python -m src.pipeline.webcam --cam               # Grad-CAM heatmap overlay
python -m src.pipeline.webcam --image face.jpg    # headless single image -> JSON + out_face.png
python -m src.pipeline.webcam --save out.mp4      # record the live session
```

Keys: `q`/`ESC` quit · `c` toggle Grad-CAM · `s` snapshot.

The webcam module runs Haar-cascade face detection, feeds crops through the
selected production model (auto device: CUDA → MPS → CPU), EMA-smooths the
probability vector, and draws label/confidence/top-3 bars/FPS on the frame.
`--image` mode is fully headless (used by the test suite).

### 4. Train Models

```bash
make train-expression          # staged RAF-DB fine-tune (ml/train.py)
```

### 5. Evaluate / Ensemble / Calibrate

```bash
make evaluate                                              # test metrics for a run
python scripts/ensemble_production.py                      # DAN+DDAMFN+ViT leaderboard, TTA, T-scaling
python scripts/ensemble_eval.py --help                     # 2-model ensemble + TTA + calibration
pytest tests/ -v                                           # full test suite
```

## Development Roadmap

- **V0** (Weeks 1-4): Research Core - Datasets, Baseline, ML Validation ✅
- **V1** (Weeks 5-8): Real-time Engine - Backend, WebSocket, Camera ✅ (webcam module)
- **V2** (Weeks 9-12): Explainability - LRP, Narrator, Event Engine
- **V3** (Weeks 13-14): Frontend - Three product modes, Dashboard
- **V4** (Weeks 15-16): Polish - Testing, Accessibility, Deployment

## License

Private - Academic Use Only

Third-party model code vendored with attribution: DAN (`yaoing/DAN`, MIT),
DDAMFN/DDAMFN++ (`SainingZhang/DDAMFN`, research use) — see `ml/models/dan.py`,
`ml/models/ddam.py`, `ml/models/mixed_feature_net.py`.
