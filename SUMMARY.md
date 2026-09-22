# AFFECTA AI — Facial Emotion Detection R&D Summary

## Objective
Build **AFFECTA AI**, a ranking-based explainable AI system for real-time facial emotion detection (LRP/XAI, AU detection, evidence-ranked explanations, product modes). Accuracy-first on **RAF-DB official test split**: ImageNet-pretrained ensemble (EffNet-B3 + ViT-Small) with strong aug + TTA, targeting **≥91% test accuracy / ≥89% Macro-F1**. Ranking/XAU/calibration are post-hoc differentiators, not the primary metric.

## Important Details
- **Emotions locked (7)** alphabetical `['angry','disgust','fear','happy','neutral','sad','surprise']`; `ACTION_UNITS=['AU4','AU6','AU7','AU9','AU10','AU12']` + heuristic `AU_TO_EMOTION`; input sizes in `ml/models/backbones.py` (`INPUT_SIZES`): efficientnet_b3=300, vit_small_patch16_224=224, resnet50=224, vgg16=224, efficientnet_b2=260. Core constants in `src/core/config.py`.
- **Compute split (user-locked)**: all training on **Kaggle (T4 GPU)**; local Mac for dataset prep, inference demo, ensemble+TTA eval, calibration, Grad-CAM/XAI, metrics analysis, docs. Original "offline/MPS-only" constraint dropped.
- **Dataset**: RAF-DB only (AffectNet skipped by user decision). Official split: train 10434 / val 1837 / test 3068 at `data/processed/rafdb/`. Source parquets `elipaluma/RAF-DB_Kaggle` at `data/raw/rafdb/`; DAN RAF-DB weights at `data/raw/dan_weights/RAF-DB.pth` (Phase 2 reference, not timm-loadable).
- **Discipline**: macro-F1 primary; test evaluated exactly once; EXP-001 invalidated (never an arm); **ADR-001 compares only official baselines EXP-004/005** (EXP-006 killed, superseded by Kaggle runs EXP-007/008); calibration is post-hoc forward-only.
- **Baseline evidence (locked tests)**:
  - EXP-004 VGG16: **test Macro-F1 66.58%, acc 67.84%**, val 67.39%; per-class F1 happy 86.5 / surprise 80.2 / disgust 68.5 / neutral 65.0 / angry 58.1 / sad 54.3 / fear 53.5; latency 14.7 ms/img p95 17.5; 1639 MiB; 134.3M params.
  - EXP-005 ResNet-50: **test Macro-F1 58.02%, acc 62.38%**, latency 7.567 ms median / 7.903 p95, 758 MiB, 23.5M params.
- **Checkpoint format**: keys `epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_val_f1, best_val_acc, history, experiment_id, config, rng_states, current_stage, stage_epoch`. Epoch checkpoints are model-ONLY; full state lives in best.pt/last.pt. **(EXP-004/005 epoch checkpoints + last.pt trimmed — only best.pt retained.)**
- Fixed staged schedule (all backbones): head_only 3 → block5 7 → block4_5 10 → block3_4_5 10; AdamW + cosine; label smoothing 0.05; class-weighted CE; patience 7 @ 0.001 on val_macro_f1 — matches the in-trainer implementation in `ml/training/train_expression.py`.
- **Kaggle credentials configured**: `~/.kaggle/kaggle.json` (chmod 600), `KaggleApi().authenticate()` → AUTH_OK as `zopevipul`. Note: `python3 -m kaggle` fails (no `__main__`) — use the Python `KaggleApi` package, not the CLI.
- ADR-001 decision criteria: Macro-F1 30%, latency 20%, acc 15%, memory/params 10%, per-class robustness 10%, XAI compatibility 10%, training cost 5%.
- **No landmark detector installed** (dlib/mediapipe/face_alignment none) — Phase 4 needs a real 68-point detector.
- **Known LRP-ε caveat**: conservation is approximate (composite z-rule; typical error 30–100% on VGG16, stable bounded maps). Exactness left for Phase 4 research budget (alpha-beta / LRP-0 first-layer).
- pytest installed per-user (`python3 -m pip install --user pytest`, 8.4.2) into `~/Library/Python/3.9`.

## Work State
### Completed
- **Project cleanup**: deleted FER2013 data (`data/raw/fer2013`, `data/processed/{train,val,test,fer2013new.csv}`, `data/splits`, `data/temp_fer2013`, `data/raw/sample`), empty raw dirs (affectnet/ckplus/expw/jaffe), dead experiments (EXP-001-VGG16, EXP-004-DIAG, EXP-004-TEST, EXP-006-EFFICIENTNET-B2, chain log), legacy scripts (all download_fer2013*/setup/convert, download_all_datasets, download_datasets, train_vgg16_clean, diagnostic, diagnostic_pretrained), `models/vgg16_best.pt`. Trimmed EXP-004/005 to best.pt only. ~9.5GB freed. Tests still pass (29 passed). Makefile + README updated to current (RAF-DB) usage.
- Kaggle credentials written + verified (`AUTH_OK`, user `zopevipul`).
- EXP-005-RESNET50 finished: Test Acc 62.38%, Macro-F1 58.02%, latency 7.567ms median / 7.903ms p95, 758 MiB, 23.5M params.
- RAF-DB prepared: `data/processed/rafdb/` train 10434 / val 1837 / test 3068 with official-split mapping (`summary.json`).
- Fixes landed: `ml/models/backbones.py` (timm loader, `INPUT_SIZES`, freeze API); `ten_crop_flip_views` (10 views); `ml/benchmarks/latency.py` DEFAULTS; `scripts/chain.py` `--next-*` passthrough; `ml/evaluate.py` local-import bug; pipeline contract (RankingEngine.compute_preliminary, uncertainty threshold 0.30).
- XAI toolchain validated on VGG16: `ml/xai/lrp.py` (Grad×Input, Grad-CAM, region aggregation), `ml/xai/lrp_epsilon.py` (composite z-rule), `ml/benchmarks/xai_probe.py` (`--method grad|lrp`) — both methods agree on brows 0.22–0.25, eyes 0.24–0.28, nose 0.15–0.18, cheeks 0.08–0.10, mouth 0.23–0.27.
- `ml/benchmarks/calibration.py` (ECE + temp via LBFGS) validated: synthetic ECE 0.255→0.027.
- Tests: **29 passed** (`tests/test_models.py`, `tests/test_tools.py`).
- timm 1.0.29 weights cached for efficientnet_b3 + vit_small_patch16_224.

### Active
- **Kaggle migration**: sync code via public GitHub (git init + `.gitignore` + commit + `gh repo create --public`) → clone on Kaggle → train EXP-007 (EffNet-B3 RAF-DB, strong aug) and EXP-008 (ViT-Small) on T4 → export ckpts/metrics to a results dataset → pull back locally.
- Pre-Kaggle code prep: add `cuda` to `ml/benchmarks/latency.py` `resolve_device`; fix `scripts/ensemble_eval.py` device selection; re-run pytest.
- Kaggle training notebook (in `notebooks/`) not yet built.
- Project dir is **not yet a git repo**; `gh` authenticated as `vizolabs`.

### Blocked
- `kaggle` CLI not on PATH (`python3 -m kaggle` has no `__main__`); use Python `KaggleApi` directly.
- No AU dataset (DISFA/BP4D, research-licensed) — `data/au` missing; `train_au.py` waits on `data/images/*.jpg` + `data/labels/*.json`.
- No landmark detector installed — needed for Phase 4 production region fidelity.
- LRP-ε exact conservation + LLM-multimodal-attribution fusion deferred to Phase 4 research budget.

## Next Move
1. Pre-Kaggle fixes: `cuda` in `ml/benchmarks/latency.py` `resolve_device`; device selection in `scripts/ensemble_eval.py`; re-run pytest.
2. `git init` + `.gitignore` (exclude `data/`, `experiments/` checkpoints, `.pytest_cache`, `*.pt`) + commit + `gh repo create <name> --public --source . --push`.
3. Build Kaggle T4 notebook: clone repo, `pip install -r requirements.txt` + timm, make RAF-DB available on Kaggle (upload via `KaggleApi` or reuse existing dataset), run EXP-007 (`train_expression.py --backbone efficientnet_b3 --experiment-id EXP-007 --batch-size 32 --data-dir data/processed/rafdb --augmentation strong`) → EXP-008 ViT-Small; export best.pt + test_metrics to results dataset.
4. Pull results back → `scripts/ensemble_eval.py` (+TTA) → calibration → Grad-CAM/XAI probe → finalize ADR-001 (winner + rationale) → product V0 pieces.

## Relevant Files
- `ml/models/backbones.py` — timm loader, `INPUT_SIZES`, freeze API; `ml/training/train_expression.py` — stage-aware trainer (no `--seed`, seed=42 via ModelConfig); `ml/training/checkpoint.py` — full-state save/load.
- `ml/benchmarks/latency.py` (needs cuda for Kaggle), `scripts/ensemble_eval.py` (ensemble + 10-view TTA; device selection needs fix).
- `ml/benchmarks/calibration.py`, `ml/xai/lrp.py`, `ml/xai/lrp_epsilon.py`, `ml/benchmarks/xai_probe.py` — Phase 4 toolchain.
- `scripts/prepare_rafdb.py` — RAF-DB → processed converter (done); `data/processed/rafdb/{train,val,test}` + `summary.json`; `data/raw/rafdb/*.parquet`.
- `scripts/chain.py`, `scripts/supervised_train.py` — local chained-training infra (superseded by Kaggle, kept for reference).
- `ml/evaluate.py` — cross-dataset eval via `--checkpoint --data-dir --split`.
- `src/models/ranking_engine.py`, `src/models/uncertainty_engine.py`, `src/core/config.py` — pipeline contract.
- `tests/test_tools.py`, `tests/test_models.py` — 29 tests passing.
- `docs/ADRS/ADR-001_BACKBONE_SELECTION.md` — PENDING (final fills after EXP-007/008).
- Results: `experiments/EXP-004-VGG16-CLEAN/` (best.pt + metrics + latency + xai_probe json), `experiments/EXP-005-RESNET50/` (best.pt + test_metrics + latency). Training for EXP-007/008 happens on Kaggle.