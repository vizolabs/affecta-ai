# Human Facial Emotion Detection

**"A Ranking-Based Explainable AI Model for Human Facial Emotion Detection"**

A real-time facial expression analysis system with evidence-ranked explainability, Grad-CAM explanations, narrator-style commentary, and single-person tracking — 100% local, frames never leave the machine.

## Headline results (RAF-DB official test split, n=3068)

| Rank | Model | Test Acc | Balanced Acc | Notes |
|------|-------|----------|--------------|-------|
| 1 | **ens_dan+ddamfn+vit** (EXP-010) | **89.99%** | 82.90% | uniform avg-probs 3-model ensemble |
| 2 | **DAN** (EXP-011) | **89.70%** | 82.75% | released ckpt, paper-exact reproduction |
| 3 | ens_dan+vit (EXP-010) | 89.31% | **83.12%** | best balanced accuracy |
| 4 | ens_dan+ddamfn (EXP-010) | 89.24% | 81.69% | ECE 0.018 (best-calibrated) |
| 5 | DDAMFN++ (EXP-009) | 83.83% | 73.40% | released ckpt, RetinaFace-realigned test |
| 6 | ViT-Small + 10-view TTA (EXP-008) | 78.88% | 73.23% | staged fine-tune (in-house) |
| 7 | EfficientNet-B3 (EXP-007) | 66.49% | 59.13 F1 | staged 30-ep schedule (in-house) |

- DAN exactly reproduces the published 89.70% (arXiv:2304.03108) on our official split.
- Temperature scaling on the champion ensemble: ECE 0.113 → 0.087 (T=0.257, val-fitted).
- TTA helps ViT (+2.25) but *hurts* DAN (89.70 → 83.83) — the 3-model ensemble without
  TTA is the best overall. Full leaderboard: `experiments/EXP-010-ENSEMBLE/production_ensemble_results.json`.
- Honest limits: the 89% macro-F1 product target was **not** met by any single in-house
  backbone (see `docs/ADRS/ADR-001`); the serving model is DAN (89.70%). Action Units are
  **untrained** (w_AU = 0, reported as such everywhere); intensity is a
  distribution-extremity proxy. Not a diagnostic tool.

## Architecture

- **Frontend**: Next.js 14 (App Router) + TypeScript + Tailwind — sci-fi login portal, Live Observer (verdict mode), Research Lab, Analytics
- **Backend**: FastAPI (Python 3.12) + JWT auth + WebSocket emotion stream
- **ML Models**: PyTorch (DAN / DDAMFN++ / ViT-Small / EfficientNet-B3)
- **Database**: SQLite (Postgres-compatible SQLAlchemy schema)
- **Real-time**: camera frames via `getUserMedia`, streamed over WebSocket (no WebRTC)
- **Explainability**: Grad-CAM served by the API; LRP implemented offline in `ml/xai` (not served)

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
│   ├── api/                # FastAPI backend (REST + WebSocket)
│   └── pipeline/           # Real-time pipeline + webcam module
├── frontend/               # Next.js web app (portal, Live, Lab, Analytics)
├── ml/                     # Training, evaluation, XAI, benchmarks
│   ├── data/               # Datasets + augmentation (incl. picklable TTA)
│   ├── models/             # backbones, production zoo (DAN, DDAMFN++), ddam
│   ├── training/           # staged trainer, checkpointing, losses
│   ├── xai/                # LRP-epsilon, Grad-CAM, region mapping
│   └── benchmarks/         # calibration (ECE/T-scaling), latency, XAI probe
├── scripts/                # prepare_rafdb, ensemble_production, report generator
├── tests/                  # pytest suite (104 tests, all passing)
├── experiments/            # EXP-004…011 results + checkpoints
├── docs/                   # numbered specs 00–22 + ADRs
└── data/                   # raw + processed RAF-DB (gitignored)
```

## Quick Start

Requires **Python 3.12** and **Node 18+**.

### 1. Install Dependencies

```bash
pip install -r requirements.txt   # opencv pinned <5 (OpenCV 5 dropped Haar cascades)
cd frontend && npm install && cd ..
```

Or in one step: `make setup`.

### 2. Data & weights (not in git)

RAF-DB and all model checkpoints are **not redistributed** in this repository:

- **RAF-DB**: request access from the official RAF-DB release (Emory University),
  then place the label/image parquets under `data/raw/rafdb/`. Prepare splits with:

  ```bash
  python scripts/prepare_rafdb.py --raw data/raw/rafdb --out data/processed/rafdb
  ```

- **Checkpoints** (~500 MB total): place the released weights at the paths expected by
  `ml/models/production.py:DEFAULT_CKPTS`:

  | Model | Expected path |
  |-------|---------------|
  | DAN (serving default) | `data/raw/dan_weights/RAF-DB.pth` |
  | DDAMFN++ | `experiments/EXP-009-DDAMFN-PP/checkpoints/rafdb_epoch20_acc0.9204_bacc0.8617.pth` |
  | ViT-Small | `experiments/EXP-008-VIT-SMALL/checkpoints/best.pt` |
  | EfficientNet-B3 | `experiments/EXP-007-EFFICIENTNET-B3/checkpoints/best.pt` |

  DAN and DDAMFN++ weights are published with their upstream repositories
  (`yaoing/DAN`, `SainingZhang/DDAMFN` — attribution below). Fine-tuned in-house
  checkpoints (ViT, B3) come from `make train-expression` / your experiment runs.

Without weights the API **still starts** and serves the UI; analysis endpoints return
503 with a pointer here, and the header chip shows "API ok" instead of "Model ready".

### 3. Run the full app (API + web)

```bash
make run            # API on http://127.0.0.1:8765 + web on http://127.0.0.1:3000
```

Open http://127.0.0.1:3000 — register a local account, then try **Live Observer**
(camera verdict mode), **Research Lab** (photo XAI) and **Analytics** (exports).
Step-by-step Windows guide: `how_to_run_the_project.txt`.

### 4. Standalone webcam module (no server required)

```bash
# Default: DAN (89.70% test acc), camera 0, EMA smoothing + HUD
python -m src.pipeline.webcam

# Options
python -m src.pipeline.webcam --model ddamfn      # DDAMFN++ (112px input)
python -m src.pipeline.webcam --cam               # Grad-CAM heatmap overlay
python -m src.pipeline.webcam --fps 5             # pace analysis at 5 fps (0 = uncapped)
python -m src.pipeline.webcam --image face.jpg    # headless single image -> JSON + out_face.png
python -m src.pipeline.webcam --save out.mp4      # record the live session
```

Keys: `q`/`ESC` quit · `c` toggle Grad-CAM · `s` snapshot.

The webcam module runs Haar-cascade face detection, feeds crops through the
selected production model (auto device: CUDA → MPS → CPU), EMA-smooths the
probability vector, and draws label/confidence/top-3 bars/FPS on the frame.
`--image` mode is fully headless (used by the test suite).

### 5. Train / Evaluate / Report

```bash
make train-expression          # staged RAF-DB fine-tune (ml/train.py)
make evaluate                  # test metrics for a run
python scripts/ensemble_production.py   # DAN+DDAMFN+ViT leaderboard, TTA, T-scaling
make report                    # regenerate Human_Facial_Emotion_Detection_Project_Report.docx
make test                      # 104 pytest tests (REST + WebSocket + ML)
```

## Development Roadmap

- **V0** (Weeks 1-4): Research Core — Datasets, Baseline, ML Validation ✅
- **V1** (Weeks 5-8): Real-time Engine — Backend, WebSocket, Camera ✅
- **V2** (Weeks 9-12): Explainability — Grad-CAM, Narrator, Event Engine ✅
- **V3** (Weeks 13-14): Frontend — Portal, Live verdict mode, Lab, Analytics ✅
- **V4** (Weeks 15-16): Polish — Testing, Accessibility, Deployment ✅

## License

Private — Academic Use Only (see `LICENSE`).

Third-party model code vendored with attribution: DAN (`yaoing/DAN`, MIT),
DDAMFN/DDAMFN++ (`SainingZhang/DDAMFN`, research use) — see `ml/models/dan.py`,
`ml/models/ddam.py`, `ml/models/mixed_feature_net.py`.
