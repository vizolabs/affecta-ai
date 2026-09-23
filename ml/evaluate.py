"""Standalone test evaluation script for AFFECTA AI.

Usage:
    python3 -m ml.evaluate --experiment-dir experiments/EXP-008-VIT-SMALL --split test
    python3 -m ml.evaluate --checkpoint experiments/EXP-008-VIT-SMALL/checkpoints/best.pt --split test
"""

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from src.core.config import EMOTIONS, NUM_EMOTIONS, PROJECT_ROOT
from src.models.expression_classifier import get_expression_model
from ml.data.dataset import create_dataloaders
from ml.training.checkpoint import (
    load_checkpoint,
    save_confusion_matrix,
    save_metrics,
)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    report = classification_report(
        all_labels, all_preds,
        target_names=EMOTIONS,
        digits=4,
        output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    
    return {
        'loss': total_loss / total,
        'accuracy': 100.0 * accuracy_score(all_labels, all_preds),
        'macro_f1': 100.0 * f1_score(all_labels, all_preds, average='macro'),
        'per_class_f1': {EMOTIONS[i]: float(per_class_f1[i]) for i in range(len(EMOTIONS))},
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def get_environment() -> dict:
    """Get environment information."""
    import platform
    import sys

    return {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': str(torch.cuda.is_available()),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test/val split')
    parser.add_argument(
        '--experiment-dir',
        type=str,
        help='Path to experiment directory (loads best.pt)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Direct path to checkpoint file'
    )
    parser.add_argument(
        '--split',
        choices=['test', 'val'],
        default='test',
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--backbone',
        choices=['vgg16', 'resnet50', 'efficientnet_b2'],
        default='vgg16',
        help='Model backbone (required if using --checkpoint without experiment dir)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for metrics JSON (default: experiment-dir/split/metrics.json)'
    )
    
    args = parser.parse_args()
    
    # Determine experiment directory and checkpoint path
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.exists():
            print(f"Error: Experiment directory not found: {exp_dir}")
            return 1
        checkpoint_path = exp_dir / 'checkpoints' / 'best.pt'
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        exp_dir = checkpoint_path.parent.parent
    else:
        print("Error: Must provide either --experiment-dir or --checkpoint")
        return 1
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    
    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Device: {device}")
    
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    backbone = config.get('backbone', args.backbone)
    experiment_id = checkpoint.get('experiment_id', 'unknown')
    
    print(f"Experiment: {experiment_id}")
    print(f"Backbone: {backbone}")
    
    # Create model
    model = get_expression_model(
        backbone=backbone,
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    from ml.training.checkpoint import load_checkpoint
    load_checkpoint(str(checkpoint_path), model, device=device)
    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"Best val F1: {checkpoint.get('best_val_f1', 'unknown'):.2f}%")
    
    # Data loaders
    image_size = 224 if backbone in ['vgg16', 'resnet50'] else 260
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=4
    )
    
    # Select loader
    if args.split == 'test':
        loader = test_loader
        split_name = 'test'
    else:
        loader = val_loader
        split_name = 'val'
    
    print(f"\nEvaluating on {split_name} set ({len(loader.dataset)} samples)...")
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {experiment_id} ({split_name})")
    print(f"{'='*60}")
    print(f"Accuracy:      {metrics['accuracy']:.2f}%")
    print(f"Macro-F1:      {metrics['macro_f1']:.2f}%")
    print(f"Loss:          {metrics['loss']:.4f}")
    print(f"\nPer-class F1:")
    for emotion, f1 in metrics['per_class_f1'].items():
        print(f"  {emotion:10s}: {f1:.2f}%")
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"{'':10s}", end="")
    for e in EMOTIONS:
        print(f"{e[:5]:>6s}", end="")
    print()
    for i, e in enumerate(EMOTIONS):
        print(f"{e:10s}", end="")
        for j in range(len(EMOTIONS)):
            print(f"{cm[i][j]:6d}", end="")
        print()
    
    # Save metrics
    output_dir = Path(args.experiment_dir) if args.experiment_dir else Path(args.checkpoint).parent.parent
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / split_name / f'{split_name}_metrics.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_metrics = {
        'experiment_id': checkpoint.get('experiment_id', 'unknown'),
        'backbone': backbone,
        'split': split_name,
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
        'checkpoint_best_val_f1': checkpoint.get('best_val_f1', 'unknown'),
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'loss': metrics['loss'],
        'per_class_f1': metrics['per_class_f1'],
        'confusion_matrix': metrics['confusion_matrix'],
        'classification_report': metrics['classification_report'],
        'environment': {
            'python': __import__('sys').version,
            'platform': __import__('platform').platform(),
            'torch': torch.__version__,
        },
        'emotions': EMOTIONS,
    }
    
    # Save metrics JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved metrics to: {output_path}")
    
    # Save confusion matrix
    if args.experiment_dir:
        from ml.training.checkpoint import save_confusion_matrix
        save_confusion_matrix(
            Path(args.experiment_dir),
            np.array(metrics['confusion_matrix']),
            EMOTIONS,
            split_name
        )
        print(f"Saved confusion matrix to: {output_dir / split_name / 'confusion_matrix.png'}")
    
    print(f"\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())