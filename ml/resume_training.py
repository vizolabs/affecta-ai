"""Resume training from checkpoint."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import sys
import platform

from src.core.config import ModelConfig, EMOTIONS, NUM_EMOTIONS, PROJECT_ROOT
from src.models.expression_classifier import get_expression_model
from ml.data.dataset import create_dataloaders, get_sample_weights


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    report = classification_report(all_labels, all_preds, target_names=EMOTIONS, digits=4, output_dict=True)
    return {
        'loss': total_loss / total,
        'accuracy': 100.0 * accuracy_score(all_labels, all_preds),
        'macro_f1': 100.0 * f1_score(all_labels, all_preds, average='macro'),
        'per_class_f1': {EMOTIONS[i]: float(f1_score(all_labels, all_preds, average=None)[i]) for i in range(len(EMOTIONS))},
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'classification_report': report,
    }


def get_environment():
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }


def resume_training(experiment_id, remaining_epochs=None):
    exp_dir = EXPERIMENTS_DIR / experiment_id
    config_path = exp_dir / "config.json"
    
    with open(config_path) as f:
        config_dict = json.load(f)
    
    config = ModelConfig(
        backbone=config_dict['backbone'],
        epochs=config_dict.get('epochs', 30),
        batch_size=config_dict.get('batch_size', 32),
        learning_rate=config_dict.get('learning_rate', 1e-4),
        weight_decay=config_dict.get('weight_decay', 1e-4),
        early_stopping_patience=config_dict.get('early_stopping_patience', 10),
        seed=config_dict.get('seed', 42),
    )
    
    if remaining_epochs is not None:
        config.epochs = remaining_epochs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {experiment_id}")
    print(f"Resuming for {config.epochs} more epochs")
    
    torch.manual_seed(config.seed)
    
    image_size = 224 if config.backbone in ['vgg16', 'resnet50'] else 260
    train_loader, val_loader, test_loader = create_dataloaders(
        'data/processed', batch_size=config.batch_size, image_size=image_size
    )
    
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    model = get_expression_model(backbone=config.backbone, pretrained=False).to(device)
    
    best_model_path = exp_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    
    class_weights = get_sample_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    best_val_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}
    
    print(f"\nTraining: {config.backbone} | Epochs: {config.epochs} | LR: {config.learning_rate}")
    print("-" * 70)
    
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        
        is_best = val_metrics['macro_f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['macro_f1']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            marker = " ★ BEST"
        else:
            patience_counter += 1
            marker = ""
        
        print(f"Epoch {epoch+1:3d}/{config.epochs} | Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.1f}% F1={val_metrics['macro_f1']:.1f}%{marker}")
        
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print("Final evaluation on test set...")
    model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
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
    
    with open(exp_dir / "test_metrics.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {experiment_id}")
    print(f"{'='*70}")
    print(f"Best Val Macro-F1:  {best_val_f1:.2f}%")
    print(f"Best Val Accuracy:  {best_val_acc:.2f}%")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.2f}%")
    print(f"Test Macro-F1:      {test_metrics['macro_f1']:.2f}%")
    print(f"\nPer-class F1:")
    for emotion, f1 in test_metrics['per_class_f1'].items():
        print(f"  {emotion:10s}: {f1:.2f}%")
    print(f"\nConfusion Matrix:")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', default='EXP-001-VGG16')
    parser.add_argument('--epochs', type=int, default=15, help='Remaining epochs to train')
    args = parser.parse_args()
    resume_training(args.experiment_id, remaining_epochs=args.epochs)
