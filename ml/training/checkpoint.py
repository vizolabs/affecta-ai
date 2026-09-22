"""Full-state checkpointing system for training."""

import torch
import numpy as np
import random
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_f1: float,
    best_val_acc: float,
    history: Dict[str, list],
    experiment_id: str,
    config: Dict[str, Any],
    rng_states: Optional[Dict[str, Any]] = None,
    stage: Optional[str] = None,
    stage_epoch: Optional[int] = None,
):
    """Save complete training checkpoint to disk.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        scheduler: Scheduler to save (optional)
        epoch: Current epoch number (global across all stages)
        best_val_f1: Best validation macro-F1 so far
        best_val_acc: Best validation accuracy so far
        history: Training history dict
        experiment_id: Experiment identifier
        config: Training configuration
        rng_states: Optional RNG states dict
        stage: Current training stage name (e.g., 'head_only', 'block5')
        stage_epoch: Epoch within the current stage
    """
    # Move model to CPU for saving (device-agnostic)
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    # Save optimizer state (move to CPU) - handle None
    optimizer_state = {}
    if optimizer is not None:
        for k, v in optimizer.state_dict().items():
            if isinstance(v, torch.Tensor):
                optimizer_state[k] = v.detach().cpu()
            elif isinstance(v, dict):
                optimizer_state[k] = {kk: vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv 
                                     for kk, vv in v.items()}
            else:
                optimizer_state[k] = v
    
    # Save scheduler state - handle None
    scheduler_state = {}
    if scheduler is not None:
        scheduler_state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                           for k, v in scheduler.state_dict().items()}
    
    # Capture RNG states if not provided
    if rng_states is None:
        rng_states = {
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            rng_states['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            try:
                rng_states['mps_rng_state'] = torch.mps.get_rng_state()
            except:
                pass
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
        'best_val_f1': best_val_f1,
        'best_val_acc': best_val_acc,
        'history': history,
        'experiment_id': experiment_id,
        'config': config,
        'rng_states': rng_states,
        'current_stage': stage or (config.get('stage') if isinstance(config, dict) else None),
        'stage_epoch': stage_epoch if stage_epoch is not None else (config.get('stage_epoch') if isinstance(config, dict) else None),
    }
    
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load complete training checkpoint from disk.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors onto
        strict: Whether to enforce strict state dict loading
    
    Returns:
        Dict with checkpoint metadata (epoch, best_val_f1, etc.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() 
                              else 'mps' if torch.backends.mps.is_available() 
                              else 'cpu')
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore RNG states (skip torch RNG for now due to serialization issues)
    if 'rng_states' in checkpoint:
        rng = checkpoint['rng_states']
        # Skip torch RNG state restoration due to serialization format issues
        # torch_state = rng.get('torch_rng_state')
        if 'numpy_rng_state' in rng:
            np.random.set_state(rng['numpy_rng_state'])
        if 'python_rng_state' in rng:
            random.setstate(rng['python_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state' in rng:
            torch.cuda.set_rng_state_all(rng['cuda_rng_state'])
        if hasattr(torch, 'mps') and torch.backends.mps.is_available() and 'mps_rng_state' in rng:
            try:
                torch.mps.set_rng_state(rng['mps_rng_state'])
            except:
                pass
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_f1': checkpoint.get('best_val_f1', 0.0),
        'best_val_acc': checkpoint.get('best_val_acc', 0.0),
        'history': checkpoint.get('history', {}),
        'experiment_id': checkpoint.get('experiment_id', 'unknown'),
        'config': checkpoint.get('config', {}),
        'current_stage': checkpoint.get('current_stage'),
        'stage_epoch': checkpoint.get('stage_epoch'),
    }


def find_latest_checkpoint(exp_dir: Path, pattern: str = "checkpoint_epoch_*.pt") -> Optional[Path]:
    """Find the latest checkpoint file in experiment directory."""
    checkpoints = list(exp_dir.glob(pattern))
    if not checkpoints:
        return None
    # Sort by epoch number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    return checkpoints[-1]


def create_experiment_structure(exp_dir: Path) -> Dict[str, Path]:
    """Create standard experiment directory structure.
    
    Returns dict of important paths.
    """
    paths = {
        'root': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'validation': exp_dir / 'validation',
        'test': exp_dir / 'test',
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def save_experiment_config(exp_dir: Path, config: Dict[str, Any], experiment_id: str):
    """Save experiment configuration."""
    config_data = {
        'experiment_id': experiment_id,
        'config': config,
    }
    import json
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)


def save_environment_info(exp_dir: Path):
    """Save environment information."""
    import platform
    import sys
    import torch
    
    env = {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'mps_available': hasattr(torch, 'mps') and torch.backends.mps.is_available(),
    }
    
    import json
    with open(exp_dir / 'environment.json', 'w') as f:
        json.dump(env, f, indent=2)


def save_metrics(exp_dir: Path, metrics: Dict[str, Any], split: str):
    """Save evaluation metrics."""
    import json
    path = exp_dir / split / f'{split}_metrics.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix(
    exp_dir: Path,
    cm: np.ndarray,
    class_names: list,
    split: str
):
    """Save confusion matrix as both JSON and PNG."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Save JSON
    cm_dict = {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
    }
    with open(exp_dir / split / 'confusion_matrix.json', 'w') as f:
        json.dump(cm_dict, f, indent=2)
    
    # Save PNG
    plt.figure(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
    except ImportError:
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.yticks(range(len(class_names)), class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{cm[i][j]}', ha='center', va='center', color='white' if cm[i][j] > cm.max() / 2 else 'black')
    plt.title(f'{split.capitalize()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(exp_dir / split / 'confusion_matrix.png', dpi=150)
    plt.close()