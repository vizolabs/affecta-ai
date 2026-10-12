# Documentation Index

Short index of the specification set (numbered specs 00–22), ADRs, and evidence figures.

## Specifications (docs/)

| # | Document |
|---|----------|
| 00 | [Master Specification](00_MASTER_SPECIFICATION.md) |
| 01 | [Product Requirements](01_PRODUCT_REQUIREMENTS.md) |
| 02 | [System Architecture](02_SYSTEM_ARCHITECTURE.md) |
| 03 | [ML Model Architecture](03_ML_MODEL_ARCHITECTURE.md) |
| 04 | [Ranking Engine](04_RANKING_ENGINE.md) |
| 05 | [Action Unit Module](05_ACTION_UNIT_MODULE.md) |
| 06 | [XAI: LRP + Grad-CAM](06_XAI_LRP_GRADCAM.md) |
| 07 | [Intensity & Uncertainty](07_INTENSITY_UNCERTAINTY.md) |
| 08 | [Dataset & Training](08_DATASET_AND_TRAINING.md) |
| 09 | [Evaluation & Ablation](09_EVALUATION_AND_ABLATION.md) |
| 10 | [Real-Time Video Pipeline](10_REALTIME_VIDEO_PIPELINE.md) |
| 11 | [Database Schema](11_DATABASE_SCHEMA.md) |
| 12 | [FastAPI Backend](12_FASTAPI_BACKEND.md) |
| 13 | [Frontend UX](13_FRONTEND_UX.md) |
| 14 | [Security & Privacy](14_SECURITY_PRIVACY.md) |
| 15 | [Reporting & Analytics](15_REPORTING_ANALYTICS.md) |
| 16 | [Deployment Infrastructure](16_DEPLOYMENT_INFRASTRUCTURE.md) |
| 17 | [Testing & QA](17_TESTING_QA.md) |
| 18 | [Development Roadmap](18_DEVELOPMENT_ROADMAP.md) |
| 19 | [Research Novelty](19_RESEARCH_NOVELTY.md) |
| 20 | [Event Engine](20_EVENT_ENGINE.md) |
| 21 | [Model Registry](21_MODEL_REGISTRY.md) |
| 22 | [Experiment Tracking](22_EXPERIMENT_TRACKING.md) |

## Architecture Decision Records (docs/ADRS/)

| ADR | Document |
|-----|----------|
| ADR-001 | [Backbone Selection](ADRS/ADR-001_BACKBONE_SELECTION.md) |
| ADR-002 | [Ranking Formula](ADRS/ADR-002_RANKING_FORMULA.md) |
| ADR-003 | [AU Subset](ADRS/ADR-003_AU_SUBSET.md) |
| ADR-004 | [LRP Frequency](ADRS/ADR-004_LRP_FREQUENCY.md) |
| ADR-005 | [Intensity Architecture](ADRS/ADR-005_INTENSITY_ARCHITECTURE.md) |
| ADR-006 | [Uncertainty Thresholds](ADRS/ADR-006_UNCERTAINTY_THRESHOLDS.md) |
| ADR-007 | [Temporal Model](ADRS/ADR-007_TEMPORAL_MODEL.md) |

## Evidence figures (docs/evidence/)

- [fig_calibration.png](evidence/fig_calibration.png) — calibration / temperature-scaling figure
- [fig_leaderboard.png](evidence/fig_leaderboard.png) — model leaderboard figure
- [kaggle_gpu_probe_2026-09-23.log](evidence/kaggle_gpu_probe_2026-09-23.log) — GPU training probe log
