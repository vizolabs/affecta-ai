# AFFECTA AI

**"A Ranking-Based Explainable AI Model for Human Facial Emotion Detection"**

A real-time facial expression analysis system with evidence-ranked explainability, Action Unit detection, LRP-based explanations, narrator-style commentary, and multi-face tracking.

## Architecture

- **Frontend**: React/Next.js + TypeScript
- **Backend**: FastAPI (Python)
- **ML Models**: PyTorch
- **Database**: PostgreSQL
- **Real-time**: WebSocket/WebRTC

## Emotion Classes

| # | Emotion |
|---|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## Project Structure

```
affecta/
├── src/                    # Application code
│   ├── core/              # Core config and utilities
│   ├── models/            # ML model implementations
│   ├── api/               # FastAPI backend
│   └── pipeline/          # Real-time processing pipeline
├── ml/                    # Training and evaluation
│   ├── data/              # Dataset loading
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation framework
│   └── xai/               # Explainable AI (LRP, Grad-CAM)
├── tests/                 # Test suite
├── data/                  # Datasets
└── notebooks/             # Training notebooks (Kaggle)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Dataset (RAF-DB)

Raw parquets live in `data/raw/rafdb/`. Prepare the train/val/test image splits:

```bash
python scripts/prepare_rafdb.py --raw data/raw/rafdb --out data/processed/rafdb
```

### 3. Train Models

```bash
# Train EfficientNet-B3 on RAF-DB
make train-expression
```

Training runs on Kaggle T4 GPU (see `notebooks/` for the training notebook).

### 4. Evaluate Models

```bash
make evaluate
```

## Model Performance Targets

| Model | Target Accuracy |
|-------|----------------|
| VGG16 Baseline | ≥ 65% |
| ResNet-50 | ≥ 67% |
| EfficientNet-B2 | ≥ 70% |

## Development Roadmap

- **V0** (Weeks 1-4): Research Core - Datasets, Baseline, ML Validation
- **V1** (Weeks 5-8): Real-time Engine - Backend, WebSocket, Camera
- **V2** (Weeks 9-12): Explainability - LRP, Narrator, Event Engine
- **V3** (Weeks 13-16): Frontend - Three product modes, Dashboard
- **V4** (Weeks 17-20): Polish - Testing, Accessibility, Deployment

## License

Private - Academic Use Only
