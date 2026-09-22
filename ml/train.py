"""Main training entry point for AFFECTA AI models."""

import argparse
import sys
from pathlib import Path

from src.core.config import ModelConfig
from ml.training.train_expression import train_expression_model


def main():
    parser = argparse.ArgumentParser(
        description='Train AFFECTA AI models'
    )
    
    parser.add_argument(
        'mode',
        choices=['expression', 'evaluate'],
        help='Training mode'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Path to data directory'
    )
    
    parser.add_argument(
        '--backbone',
        choices=['vgg16', 'resnet50', 'efficientnet_b2'],
        default='vgg16',
        help='Model backbone'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        default='EXP-004-VGG16-CLEAN',
        help='Experiment ID'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--split',
        choices=['test', 'val'],
        default='test',
        help='Split to evaluate (for evaluate mode)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint path (for evaluate mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'expression':
        print("="*60)
        print("AFFECTA AI - Expression Classifier Training")
        print("="*60)
        print(f"Backbone: {args.backbone}")
        print(f"Experiment: {args.experiment_id}")
        print(f"Data: {args.data_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch: {args.batch_size}")
        print(f"LR: {args.lr}")
        print("="*60)
        
        config = ModelConfig(
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            early_stopping_patience=7,
        )
        
        results = train_expression_model(
            data_dir=args.data_dir,
            config=config,
            experiment_id=args.experiment_id,
            resume_from=args.resume
        )
        
        print(f"\nTraining complete!")
        print(f"Best Val Macro-F1: {results['best_val_macro_f1']:.2f}%")
        print(f"Test Macro-F1: {results['test_macro_f1']:.2f}%")
        
    elif args.mode == 'evaluate':
        from ml.evaluate import main as eval_main
        # Pass args to evaluate script
        sys.argv = [
            'evaluate.py',
            '--experiment-dir', args.experiment_id if args.experiment_id else '',
            '--checkpoint', args.checkpoint or '',
            '--split', args.split,
            '--batch-size', str(args.batch_size),
        ]
        eval_main()


if __name__ == "__main__":
    main()