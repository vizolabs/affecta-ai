"""Training callbacks: early stopping, SWA, LR monitoring.

EarlyStopping matches the in-trainer implementation used by EXP-004/005
(patience on val_macro_f1, restore_best) and is re-exported for reuse.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping with validation metric monitoring."""

    def __init__(
        self,
        monitor: str = 'val_macro_f1',
        mode: str = 'max',
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best: bool = True,
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


class ModelAveragingSWA:
    """Stochastic Weight Averaging.

    Call update(model) after each epoch once epoch >= start_epoch; at the end,
    apply_to(model) writes averaged weights into the live model (BN stats are
    left as-is unless update_bn is run via copy + BN refresh).
    """

    def __init__(self, start_epoch: int = 20):
        self.start_epoch = start_epoch
        self.n_averaged = 0
        self.swa_state: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module, epoch: int) -> bool:
        if epoch < self.start_epoch:
            return False
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self.swa_state is None:
            self.swa_state = state
            self.n_averaged = 1
        else:
            for k, v in state.items():
                if torch.is_floating_point(self.swa_state[k]):
                    self.swa_state[k].mul_(self.n_averaged).add_(v).div_(self.n_averaged + 1)
                else:
                    self.swa_state[k] = v
            self.n_averaged += 1
        return True

    def apply_to(self, model: nn.Module) -> int:
        if self.swa_state is None:
            return 0
        model.load_state_dict(self.swa_state)
        return self.n_averaged

    def state_dict(self) -> dict:
        return {
            'start_epoch': self.start_epoch,
            'n_averaged': self.n_averaged,
            'swa_state': self.swa_state,
        }

    def load_state_dict(self, sd: dict):
        self.start_epoch = sd.get('start_epoch', self.start_epoch)
        self.n_averaged = sd.get('n_averaged', 0)
        self.swa_state = sd.get('swa_state')


class LRMonitor:
    """Records optimizer LR per step into a flat history list."""

    def __init__(self):
        self.history: List[float] = []

    def record(self, optimizer: torch.optim.Optimizer) -> float:
        lr = optimizer.param_groups[0]['lr']
        self.history.append(lr)
        return lr
