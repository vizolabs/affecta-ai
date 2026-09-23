"""Train expression classifier with staged transfer learning and full checkpointing.

Architecture:
    - Data-driven training stages (head_only -> block5 -> block4_5 -> block3_4_5)
    - Stage-aware resume: checkpoints store current_stage + stage_epoch so a
      resumed run continues the exact stage/epoch (never restarts Stage 1)
    - Full-state checkpoints in best.pt / last.pt (model + optimizer + scheduler
      + history + stage info); lightweight model-only epoch checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from src.core.config import ModelConfig, EMOTIONS, NUM_EMOTIONS, PROJECT_ROOT
from src.models.expression_classifier import get_expression_model
from ml.data.dataset import create_dataloaders, get_sample_weights
from ml.data.augmentation import build_train_transform, mixup_batch, cutmix_batch
from ml.training.losses import build_criterion, mixup_criterion
from ml.training.callbacks import ModelAveragingSWA, LRMonitor
from ml.models.backbones import build_expression_model, input_size_for
from ml.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    create_experiment_structure,
    save_experiment_config,
    save_environment_info,
    save_metrics,
    save_confusion_matrix,
)


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ---------------------------------------------------------------------------
# Training stages (data-driven)
#   VGG16 block indices (0-indexed): 0=block1 ... 4=block5
# ---------------------------------------------------------------------------
STAGES = [
    {
        'name': 'head_only',
        'max_epochs': 3,
        'unfreeze_blocks': [],       # backbone fully frozen
        'head_lr': 3e-4,
        'backbone_lr': 3e-4,         # unused in head-only stage
        'T_max': 3,
        'description': 'Head only (backbone frozen)',
    },
    {
        'name': 'block5',
        'max_epochs': 7,
        'unfreeze_blocks': [4],      # block5
        'head_lr': 1e-4,
        'backbone_lr': 1e-5,
        'T_max': 7,
        'description': 'Fine-tune block5 + head',
    },
    {
        'name': 'block4_5',
        'max_epochs': 10,
        'unfreeze_blocks': [3, 4],   # blocks4-5
        'head_lr': 1e-4,
        'backbone_lr': 1e-5,
        'T_max': 10,
        'description': 'Fine-tune blocks 4-5 + head',
    },
    {
        'name': 'block3_4_5',
        'max_epochs': 10,
        'unfreeze_blocks': [2, 3, 4],  # blocks3-5
        'head_lr': 1e-4,
        'backbone_lr': 1e-5,
        'T_max': 10,
        'description': 'Fine-tune blocks 3-5 + head',
    },
]

STAGE_NAME_TO_IDX = {s['name']: i for i, s in enumerate(STAGES)}


class EarlyStopping:
    """Early stopping with validation metric monitoring."""

    def __init__(
        self,
        monitor: str = "val_macro_f1",
        mode: str = "max",
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best: bool = True
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        improved = False
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                improved = True
        else:
            if score < self.best_score - self.min_delta:
                improved = True

        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def compute_class_weights(loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    class_counts = np.zeros(num_classes)
    for _, labels in loader:
        for label in labels.numpy():
            class_counts[label] += 1

    # Inverse frequency weighting - normalized to mean 1.0
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def prune_epoch_checkpoints(checkpoints_dir, keep: int = 5):
    """Delete old epoch checkpoints, keeping only the newest `keep`."""
    import re
    checkpoint_files = sorted(
        Path(checkpoints_dir).glob('checkpoint_epoch_*.pt'),
        key=lambda p: int(re.search(r'(\d+)', p.stem).group(1))
    )
    for old in checkpoint_files[:-keep]:
        old.unlink(missing_ok=True)


def log_class_distribution(loader: DataLoader, prefix: str = ""):
    """Log class distribution in a dataset."""
    class_counts = np.zeros(NUM_EMOTIONS)
    for _, labels in loader:
        for label in labels.numpy():
            class_counts[label] += 1

    total = class_counts.sum()
    print(f"\n{prefix}Class Distribution:")
    for i, emotion in enumerate(EMOTIONS):
        print(f"  {emotion:10s}: {int(class_counts[i]):5d} ({class_counts[i]/total*100:.1f}%)")


def log_predicted_distribution(predictions: np.ndarray, prefix: str = ""):
    """Log predicted class distribution."""
    pred_counts = np.bincount(predictions, minlength=NUM_EMOTIONS)
    print(f"\n{prefix}Predicted Distribution:")
    for i, emotion in enumerate(EMOTIONS):
        print(f"  {emotion:10s}: {pred_counts[i]:5d} ({pred_counts[i]/len(predictions)*100:.1f}%)")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    grad_accum: int = 1,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    step = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_mix = mixup_alpha > 0 or cutmix_alpha > 0
        if use_mix:
            if cutmix_alpha > 0 and mixup_alpha <= 0:
                images, t_a, t_b, lam = cutmix_batch(images, labels, cutmix_alpha)
            elif mixup_alpha > 0 and cutmix_alpha <= 0:
                images, t_a, t_b, lam = mixup_batch(images, labels, mixup_alpha)
            else:
                # both enabled: randomly choose per batch
                if torch.rand(1).item() < 0.5:
                    images, t_a, t_b, lam = mixup_batch(images, labels, mixup_alpha)
                else:
                    images, t_a, t_b, lam = cutmix_batch(images, labels, cutmix_alpha)

        with torch.autocast(autocast_device, enabled=use_amp):
            outputs = model(images)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, t_a, t_b, lam) / grad_accum
            else:
                loss = criterion(outputs, labels) / grad_accum

        scaler.scale(loss).backward()
        step += 1
        if step % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if not use_mix:
            correct += predicted.eq(labels).sum().item()
        else:
            correct += (predicted.eq(t_a) | predicted.eq(t_b)).sum().item()

    return {
        'loss': total_loss / max(total, 1),
        'accuracy': 100.0 * correct / max(total, 1),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    log_dist: bool = True
) -> Dict[str, Any]:
    model.eval()
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if log_dist:
        log_predicted_distribution(all_preds, "Val ")

    report = classification_report(
        all_labels, all_preds,
        target_names=EMOTIONS,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return {
        'accuracy': 100.0 * accuracy_score(all_labels, all_preds),
        'macro_f1': 100.0 * f1_score(all_labels, all_preds, average='macro'),
        'per_class_f1': {EMOTIONS[i]: float(per_class_f1[i]) for i in range(len(EMOTIONS))},
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
    }


def get_environment() -> Dict[str, str]:
    import platform
    import sys
    import torch

    return {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': str(torch.cuda.is_available()),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'mps_available': str(hasattr(torch, 'mps') and torch.backends.mps.is_available()),
    }


def get_pin_memory() -> bool:
    """Return False for MPS, True for CUDA/CPU."""
    return not (hasattr(torch, 'mps') and torch.backends.mps.is_available())


class StageTrainer:
    """Staged transfer-learning trainer with stage-aware resume."""

    def __init__(self, model, stages, device):
        self.model = model
        self.stages = stages
        self.device = device

    def prepare_for_stage(self, stage: Dict[str, Any]) -> optim.Optimizer:
        """Set freeze state and build optimizer for a stage."""
        stage_name = stage['name']
        unfreeze_blocks = stage['unfreeze_blocks']

        if stage_name == 'head_only':
            if hasattr(self.model, 'freeze_backbone'):
                self.model.freeze_backbone()
            print("Backbone frozen. Head trainable.")
            optimizer = optim.AdamW(
                self.model.head.parameters(),
                lr=stage['head_lr'],
                weight_decay=1e-4
            )
        else:
            # Cumulative unfreeze: freeze all, then unfreeze requested blocks
            if hasattr(self.model, 'freeze_backbone'):
                self.model.freeze_backbone()
            backbone_params = []
            for block_idx in unfreeze_blocks:
                self.model.unfreeze_block(block_idx)
                backbone_params += list(self.model.features.blocks[block_idx].parameters())

            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': stage['backbone_lr']},
                {'params': self.model.head.parameters(), 'lr': stage['head_lr']},
            ], weight_decay=1e-4)

        return optimizer


def train_expression_model(
    data_dir: str,
    config: ModelConfig = None,
    experiment_id: str = "EXP-004-VGG16-CLEAN",
    resume_from: Optional[str] = None,
    max_stage_idx: Optional[int] = None,
    stages: Optional[List[Dict[str, Any]]] = None,
    diag_limit: Optional[int] = None,
    extend_last_stage: Optional[int] = None,
    augmentation: str = 'baseline',
    loss_name: str = 'ce',
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    grad_accum: int = 1,
    use_amp: bool = False,
    swa_start: Optional[int] = None,
    init_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Train expression classifier using staged transfer learning.

    Args:
        data_dir: Path to processed data dir (train/val/test folders)
        config: ModelConfig
        experiment_id: Experiment identifier
        resume_from: Path to a checkpoint to resume from (full stage resume)
        max_stage_idx: Maximum stage index to run (for diagnostics, default = all)
        stages: Stage definitions (default = STAGES)
        diag_limit: If set, cap train set size for quick diagnostic runs
        augmentation: 'baseline' (legacy weak) or 'strong' (RandAugment etc.)
        loss_name: 'ce' or 'focal'
        mixup_alpha: MixUp beta alpha (0 = off)
        cutmix_alpha: CutMix beta alpha (0 = off)
        grad_accum: Gradient accumulation steps
        use_amp: Mixed precision (CUDA only in practice)
        swa_start: Global epoch to begin SWA averaging (None = off)
        init_weights: Optional checkpoint path for warm-start (e.g. DAN RAF-DB)
    """
    if config is None:
        config = ModelConfig()

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    pin_memory = get_pin_memory()

    print(f"Device: {device} (pin_memory={pin_memory})")
    print(f"Experiment: {experiment_id}")
    print(f"Backbone: {config.backbone}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    stages = stages or STAGES
    if max_stage_idx is not None:
        stages = stages[:max_stage_idx]
    if extend_last_stage is not None:
        last = stages[-1]
        last_pre = dict(last)
        last['max_epochs'] = extend_last_stage
        last['T_max'] = max(extend_last_stage, 1)
        print(f"Extended last stage '{last['name']}': epochs {last_pre['max_epochs']} -> {extend_last_stage}")

    # Create experiment directory structure
    exp_dir = EXPERIMENTS_DIR / experiment_id
    paths = create_experiment_structure(exp_dir)

    # Save config and environment
    config_dict = {
        'backbone': config.backbone,
        'num_classes': config.num_classes,
        'pretrained': config.pretrained,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'batch_size': config.batch_size,
        'early_stop_patience': config.early_stopping_patience,
        'early_stop_monitor': 'val_macro_f1',
        'early_stop_min_delta': 0.001,
        'seed': config.seed,
        'emotions': EMOTIONS,
        'stages': [
            {'name': s['name'], 'epochs': s['max_epochs'], 'unfreeze': s['unfreeze_blocks']}
            for s in stages
        ],
        'head_lr': 3e-4,
        'backbone_lr': 1e-5,
        'label_smoothing': 0.05,
        'use_class_weights': True,
        'augmentation': augmentation,
        'loss_name': loss_name,
        'mixup_alpha': mixup_alpha,
        'cutmix_alpha': cutmix_alpha,
        'grad_accum': grad_accum,
        'use_amp': use_amp,
        'swa_start': swa_start,
        'init_weights': init_weights,
    }
    save_experiment_config(exp_dir, config_dict, experiment_id)
    save_environment_info(exp_dir)

    # Create data loaders
    image_size = input_size_for(config.backbone)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=config.batch_size,
        image_size=image_size,
        num_workers=4,
        pin_memory=False,  # MPS doesn't support pin_memory
        augmentation=augmentation,
    )

    # Diagnostic: cap train set if requested (stratified across classes)
    if diag_limit is not None:
        samples = train_loader.dataset.samples
        per_class = max(1, diag_limit // NUM_EMOTIONS)
        stratified = []
        for c in range(NUM_EMOTIONS):
            c_samples = [s for s in samples if s[1] == c][:per_class]
            stratified.extend(c_samples)
        train_loader.dataset.samples = stratified
        print(f"\n[DIAG] Training on {len(train_loader.dataset)} samples "
              f"(~{per_class}/class, stratified)")

    print(f"\nTrain: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    log_class_distribution(train_loader, "Train ")
    log_class_distribution(val_loader, "Val ")

    # Class weights
    class_weights = compute_class_weights(train_loader, NUM_EMOTIONS)
    print(f"\nClass Weights:")
    for i, emotion in enumerate(EMOTIONS):
        print(f"  {emotion:10s}: {class_weights[i]:.3f}")

    # Criterion (shared across stages)
    criterion = build_criterion(
        loss_name=loss_name,
        class_weights=class_weights,
        label_smoothing=0.05,
    ).to(device)

    # Create model (torchvision baselines or timm via unified factory)
    model = build_expression_model(
        backbone=config.backbone,
        pretrained=config.pretrained,
        dropout=0.5,
        init_weights=init_weights,
    ).to(device)

    # Training state
    best_val_f1 = 0.0
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': [],
        'stage': [], 'epoch_type': [],
    }
    swa = ModelAveragingSWA(start_epoch=swa_start) if swa_start is not None else None
    lr_monitor = LRMonitor()

    resume_stage_idx = 0
    resume_stage_epoch = 0

    # ---------------- Resume handling ----------------
    if resume_from:
        print(f"Resuming from {resume_from}")
        checkpoint_data = load_checkpoint(resume_from, model, device=torch.device('cpu'))
        best_val_f1 = checkpoint_data.get('best_val_f1', 0.0)
        best_val_acc = checkpoint_data.get('best_val_acc', 0.0)
        history = checkpoint_data.get('history', history)
        saved_stage = checkpoint_data.get('current_stage')
        saved_stage_epoch = checkpoint_data.get('stage_epoch', 0)

        if saved_stage in STAGE_NAME_TO_IDX:
            resume_stage_idx = STAGE_NAME_TO_IDX[saved_stage]
            resume_stage_epoch = saved_stage_epoch
            # If we just completed a stage (stage_epoch == max_epochs), move to next stage
            if resume_stage_epoch >= stages[resume_stage_idx]['max_epochs']:
                resume_stage_idx += 1
                resume_stage_epoch = 0
        else:
            resume_stage_idx = 0
            resume_stage_epoch = 0

        print(f"Resumed: stage={saved_stage} (idx {resume_stage_idx}), "
              f"stage_epoch={resume_stage_epoch}, best F1={best_val_f1:.2f}%")

    # ---------------- Staged training loop ----------------
    global_epoch = len(history['train_loss'])

    for stage_idx, stage in enumerate(stages):
        if stage_idx < resume_stage_idx:
            print(f"\nSkipping completed stage: {stage['name']}")
            continue

        print(f"\n{'='*70}")
        print(f"STAGE {stage_idx+1}: {stage['name']} - {stage['description']}")
        print(f"{'='*70}")

        optimizer = StageTrainer(model, stages, device).prepare_for_stage(stage)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage['T_max'])

        # Load optimizer/scheduler state if resuming mid-stage
        if stage_idx == resume_stage_idx and resume_stage_epoch > 0:
            if resume_from:
                load_checkpoint(resume_from, model, optimizer, scheduler, device=torch.device('cpu'))
                print(f"Loaded optimizer/scheduler state for {stage['name']} (epoch {resume_stage_epoch})")

        early_stopping = EarlyStopping(
            monitor='val_macro_f1',
            mode='max',
            patience=config.early_stopping_patience,
            min_delta=0.001,
            restore_best=True
        )

        start_epoch = resume_stage_epoch if stage_idx == resume_stage_idx else 0

        for stage_epoch_count in range(start_epoch, stage['max_epochs']):
            lr_monitor.record(optimizer)
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                grad_accum=grad_accum,
                use_amp=use_amp,
            )
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()
            if swa is not None:
                swa.update(model, global_epoch)

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(0.0)  # no criterion on val; kept for shape
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_macro_f1'].append(val_metrics['macro_f1'])
            history['stage'].append(stage['name'])
            history['epoch_type'].append(stage['name'])

            is_best = val_metrics['macro_f1'] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics['macro_f1']
                best_val_acc = val_metrics['accuracy']

            marker = " ★" if is_best else ""
            print(
                f"Epoch {global_epoch+1:3d} [{stage['name']}] "
                f"{stage_epoch_count+1}/{stage['max_epochs']} | "
                f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.1f}% | "
                f"Val Acc={val_metrics['accuracy']:.1f}% F1={val_metrics['macro_f1']:.1f}%{marker}"
            )

            # Lightweight epoch checkpoint (model state + stage info only)
            save_checkpoint(
                paths['checkpoints'] / f'checkpoint_epoch_{global_epoch+1:03d}.pt',
                model, None, None,
                global_epoch, best_val_f1, best_val_acc, history,
                experiment_id, config_dict,
                stage=stage['name'],
                stage_epoch=stage_epoch_count + 1
            )
            # Rolling retention: keep only the newest KEEP_N epoch checkpoints
            # to bound disk usage (full state lives in best.pt / last.pt).
            prune_epoch_checkpoints(paths['checkpoints'], keep=5)

            # Full-state best checkpoint
            if is_best:
                save_checkpoint(
                    paths['checkpoints'] / 'best.pt',
                    model, optimizer, scheduler,
                    global_epoch, best_val_f1, best_val_acc, history,
                    experiment_id, config_dict,
                    stage=stage['name'],
                    stage_epoch=stage_epoch_count + 1
                )

            global_epoch += 1

            if early_stopping(val_metrics['macro_f1'], model):
                print(f"Early stopping at {stage['name']} epoch {stage_epoch_count+1}")
                break

        # Restore best weights within this stage if available
        if early_stopping.best_state is not None:
            model.load_state_dict(early_stopping.best_state)
            print(f"Restored best {stage['name']} model (Val F1: {early_stopping.best_score:.2f}%)")

        # Full-state last checkpoint at end of stage
        save_checkpoint(
            paths['checkpoints'] / 'last.pt',
            model, optimizer, scheduler,
            global_epoch - 1, best_val_f1, best_val_acc, history,
            experiment_id, config_dict,
            stage=stage['name'],
            stage_epoch=stage['max_epochs']
        )
        print(f"Saved full-state last.pt after {stage['name']} (epochs done: {stage['max_epochs']})")

    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    print(f"\n{'='*70}")
    print("Final evaluation on test set...")

    # Optionally swap in SWA-averaged weights before final eval
    if swa is not None and swa.n_averaged > 0:
        n = swa.apply_to(model)
        print(f"Applied SWA weights (averaged over {n} snapshots)")

    # Load best model (by val Macro-F1) unless SWA is active
    best_path = paths['checkpoints'] / 'best.pt'
    if swa is None and best_path.exists():
        load_checkpoint(best_path, model, device=torch.device('cpu'))
        print(f"Loaded best model from {best_path}")

    test_metrics = evaluate(model, test_loader, device, log_dist=True)

    # Save final results
    final_results = {
        'experiment_id': experiment_id,
        'backbone': config.backbone,
        'best_val_macro_f1': best_val_f1,
        'best_val_acc': best_val_acc,
        'test_acc': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_per_class_f1': test_metrics['per_class_f1'],
        'test_confusion_matrix': test_metrics['confusion_matrix'],
        'test_classification_report': test_metrics['classification_report'],
        'history': history,
        'config': config_dict,
        'environment': get_environment(),
        'timestamp': datetime.now().isoformat(),
        'emotions': EMOTIONS,
    }

    save_metrics(exp_dir, test_metrics, 'test')
    save_confusion_matrix(exp_dir, np.array(test_metrics['confusion_matrix']), EMOTIONS, 'test')

    with open(exp_dir / 'test_metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    with open(exp_dir / 'train_history.json', 'w') as f:
        json.dump({
            'history': history,
            'best_val_f1': best_val_f1,
            'best_val_acc': best_val_acc,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS: {experiment_id}")
    print(f"{'='*70}")
    print(f"Best Val Macro-F1:  {best_val_f1:.2f}%")
    print(f"Best Val Accuracy:  {best_val_acc:.2f}%")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.2f}%")
    print(f"Test Macro-F1:      {test_metrics['macro_f1']:.2f}%")
    print(f"\nPer-class Test F1:")
    for emotion, f1 in test_metrics['per_class_f1'].items():
        print(f"  {emotion:10s}: {f1:.2f}%")
    print(f"\nConfusion Matrix (Test):")
    cm = np.array(test_metrics['confusion_matrix'])
    print(f"{'':10s}", end="")
    for e in EMOTIONS:
        print(f"{e[:5]:>6s}", end="")
    print()
    for i, e in enumerate(EMOTIONS):
        print(f"{e:10s}", end="")
        for j in range(len(EMOTIONS)):
            print(f"{cm[i][j]:6d}", end="")
        print()
    print(f"\nSaved to: {exp_dir}")
    print(f"{'='*70}")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train expression classifier')
    parser.add_argument('--backbone', default='vgg16',
                        choices=['vgg16', 'resnet50', 'efficientnet_b2',
                                 'efficientnet_b3', 'efficientnet_b4',
                                 'vit_small_patch16_224', 'vit_base_patch16_224'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--experiment-id', default='EXP-004-VGG16-CLEAN')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max-stage', type=int, default=None, help='Only run up to stage index (diagnostics)')
    parser.add_argument('--diag-limit', type=int, default=None, help='Cap training samples for diagnostic runs')
    parser.add_argument('--extend-last-stage', type=int, default=None, help='Continue last stage for N total epochs (resume use)')
    parser.add_argument('--augmentation', default='baseline', choices=['baseline', 'strong'])
    parser.add_argument('--loss', default='ce', choices=['ce', 'focal'])
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--swa-start', type=int, default=None)
    parser.add_argument('--init-weights', type=str, default=None,
                        help='Warm-start checkpoint (e.g. DAN RAF-DB weights)')
    args = parser.parse_args()

    config = ModelConfig(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=7,
    )

    train_expression_model(
        args.data_dir, config, args.experiment_id, args.resume,
        max_stage_idx=args.max_stage,
        diag_limit=args.diag_limit,
        extend_last_stage=args.extend_last_stage,
        augmentation=args.augmentation,
        loss_name=args.loss,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        grad_accum=args.grad_accum,
        use_amp=args.amp,
        swa_start=args.swa_start,
        init_weights=args.init_weights,
    )