"""Training losses: Focal, LabelSmoothingCE, and MixUp/CutMix-aware CE wrapper.

Defaults preserve EXP-004/005 behavior: class-weighted CE + label_smoothing=0.05.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """CrossEntropy with label smoothing and optional class weights."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.05,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.smoothing,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """Focal loss (Lin et al.) with optional label smoothing and class weights.

    gamma=0 reduces to (smoothed) CE; use gamma=2.0 for hard-example emphasis.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.smoothing,
            reduction='none',
        )
        if self.gamma <= 0:
            focal = ce
        else:
            pt = torch.exp(-ce)
            focal = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


class SoftTargetCE(nn.Module):
    """CE for soft targets produced by MixUp/CutMix (row-wise log-prob dot product)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return (-targets * log_probs).sum(dim=1).mean()


def build_criterion(
    loss_name: str = 'ce',
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.05,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory used by the trainer.

    loss_name: 'ce' (default) | 'focal'
    """
    if class_weights is not None:
        class_weights = class_weights.float()
    if loss_name == 'ce':
        if label_smoothing > 0:
            return LabelSmoothingCE(weight=class_weights, smoothing=label_smoothing)
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == 'focal':
        return FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            smoothing=label_smoothing,
        )
    raise ValueError(f'Unknown loss: {loss_name}')


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Weighted CE for MixUp/CutMix pairs; falls back cleanly when lam == 1."""
    if lam >= 1.0:
        return criterion(logits, target_a)
    return lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
