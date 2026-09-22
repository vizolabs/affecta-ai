"""Model evaluation framework."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from src.core.config import EMOTIONS, ACTION_UNITS


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(predictions, labels):
        cm[label][pred] += 1
    return cm


def compute_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """Compute metrics from confusion matrix.
    
    Returns:
        Dict with accuracy, precision, recall, f1 per class
    """
    num_classes = cm.shape[0]
    
    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) \
            if (precision[i] + recall[i]) > 0 else 0
    
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision_per_class': dict(zip(EMOTIONS, precision.tolist())),
        'recall_per_class': dict(zip(EMOTIONS, recall.tolist())),
        'f1_per_class': dict(zip(EMOTIONS, f1.tolist())),
        'macro_precision': float(np.mean(precision)),
        'macro_recall': float(np.mean(recall)),
        'macro_f1': float(np.mean(f1)),
    }


def evaluate_expression_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate expression classifier on test set.
    
    Returns:
        Evaluation metrics dict
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(all_preds, all_labels, len(EMOTIONS))
    
    # Compute metrics
    metrics = compute_metrics_from_cm(cm)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_au_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute AU detection metrics.
    
    Args:
        predictions: Binary predictions [N, num_aus]
        labels: Binary labels [N, num_aus]
    
    Returns:
        AU metrics dict
    """
    num_aus = predictions.shape[1]
    
    precision = np.zeros(num_aus)
    recall = np.zeros(num_aus)
    f1 = np.zeros(num_aus)
    
    for i in range(num_aus):
        tp = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum()
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) \
            if (precision[i] + recall[i]) > 0 else 0
    
    accuracy = (predictions == labels).mean()
    
    return {
        'accuracy': float(accuracy),
        'precision_per_au': dict(zip(ACTION_UNITS, precision.tolist())),
        'recall_per_au': dict(zip(ACTION_UNITS, recall.tolist())),
        'f1_per_au': dict(zip(ACTION_UNITS, f1.tolist())),
        'mean_f1': float(np.mean(f1)),
    }
