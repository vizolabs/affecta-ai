"""Confidence calibration utilities: ECE/MCE and temperature scaling.

Post-hoc calibration operates on already-logged logits (forward pass only) and
never retrains or modifies the checkpoint, preserving the test-once discipline.

Usage (Phase 4):
    logits, targets = my_eval_harness()      # numpy [N, C], int64 [N]
    ece = compute_ece(logits, targets)       # uncalibrated ECE
    temp = fit_temperature(val_logits, val_targets)
    ece_cal = compute_ece(apply_temperature(logits, temp), targets)
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_ece(
    logits: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Expected Calibration Error (ECE) and MCE, plus confidence histogram.

    logits: shape [N, C]; probs_softmax applied internally.
    targets: shape [N,].
    """
    probs = _softmax(logits)
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == targets).astype(float)
    ece = 0.0
    mce = 0.0
    counts = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confs >= lo) & (confs < hi)
        n = int(mask.sum())
        counts.append(n)
        if n == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = accs[mask].mean()
        ece += (n / len(confs)) * abs(bin_acc - bin_conf)
        mce = max(mce, abs(bin_acc - bin_conf))
    return {
        'ece': float(ece),
        'mce': float(mce),
        'n_samples': int(len(confs)),
        'bin_counts': counts,
        'avg_confidence': float(confs.mean()),
        'accuracy': float(accs.mean()),
    }


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    init: float = 1.0,
    lr: float = 0.01,
    steps: int = 500,
) -> float:
    """Optimise temperature T by minimising cross-entropy of logits/T."""
    logits_t = torch.from_numpy(logits.astype(np.float32))
    targets_t = torch.from_numpy(targets.astype(np.int64))
    T = torch.tensor(init, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=steps)
    loss_full = None

    def closure():
        nonlocal loss_full
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T, targets_t)
        loss.backward()
        loss_full = float(loss.item())
        return loss

    opt.step(closure)
    return float(T.detach().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated probabilities (softmax(logits / T))."""
    return _softmax(np.asarray(logits, dtype=np.float32) / float(temperature))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)