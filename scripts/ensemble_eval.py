"""Ensemble evaluation with optional 5-crop+flip TTA.

Loads N checkpoints (same dataset layout), averages softmax probabilities
across models and TTA views, reports accuracy / Macro-F1 / ECE.

Usage:
    python scripts/ensemble_eval.py \
        --checkpoints experiments/EXP-007/checkpoints/best.pt,experiments/EXP-008/checkpoints/best.pt \
        --backbones efficientnet_b3,dan_efficientnet_b0 \
        --data-dir data/processed/rafdb --split test \
        --tta --output experiments/ensemble_rafdb_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from src.core.config import EMOTIONS, PROJECT_ROOT
from ml.data.augmentation import build_eval_transform, ten_crop_flip_views
from ml.models.backbones import build_expression_model, input_size_for
from ml.training.checkpoint import load_checkpoint
from ml.benchmarks.calibration import compute_ece, fit_temperature


class FolderDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, transform):
        self.transform = transform
        self.samples = []
        root = data_dir / split
        for i, emotion in enumerate(EMOTIONS):
            ed = root / emotion
            if not ed.exists():
                continue
            for p in sorted(list(ed.glob('*.jpg')) + list(ed.glob('*.png'))):
                self.samples.append((str(p), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def load_model(ckpt_path: str, backbone: str, device: torch.device) -> torch.nn.Module:
    size = input_size_for(backbone)
    model = build_expression_model(backbone, pretrained=False, dropout=0.5)
    load_checkpoint(ckpt_path, model, device=torch.device('cpu'))
    model.to(device)
    model.eval()
    return model, size


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
    image_size: int,
) -> np.ndarray:
    all_logits = []
    for batch in loader:
        images, _ = batch
        if tta:
            views = images  # (B, 10, 3, S, S)
            b, n, c, h, w = views.shape
            logits = model(views.reshape(b * n, c, h, w).to(device))
            logits = logits.view(b, n, -1).mean(dim=1)
        else:
            logits = model(images.to(device))
        all_logits.append(logits.detach().float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
    image_size: int,
) -> np.ndarray:
    logits = predict_logits(model, loader, device, tta, image_size)
    return _softmax(logits)


class TTADataset(Dataset):
    """Yields (views_tensor 10x3xSxS, label) using ten_crop_flip_views."""

    def __init__(self, data_dir: Path, split: str, image_size: int):
        self.view_fn = ten_crop_flip_views(image_size)
        self.samples = []
        root = data_dir / split
        for i, emotion in enumerate(EMOTIONS):
            ed = root / emotion
            if not ed.exists():
                continue
            for p in sorted(list(ed.glob('*.jpg')) + list(ed.glob('*.png'))):
                self.samples.append((str(p), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.view_fn(img), label


def main():
    parser = argparse.ArgumentParser(description='Ensemble + TTA evaluation')
    parser.add_argument('--checkpoints', required=True, help='Comma-separated checkpoint paths')
    parser.add_argument('--backbones', required=True, help='Comma-separated backbone names (same order)')
    parser.add_argument('--data-dir', default=str(PROJECT_ROOT / 'data/processed/rafdb'))
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--tta', action='store_true', help='10-view 5-crop+flip TTA')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for calibration')
    parser.add_argument('--calibrate-on', default='val', choices=['train', 'val', 'test', 'none'],
                        help='Split used to fit temperature scaling; "none" skips calibration (T=1)')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    ckpts = [c.strip() for c in args.checkpoints.split(',') if c.strip()]
    backbones = [b.strip() for b in args.backbones.split(',') if b.strip()]
    if len(ckpts) != len(backbones):
        raise SystemExit('--checkpoints and --backbones must have the same length')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    data_dir = Path(args.data_dir)

    def run_split(split: str, tta: bool) -> tuple:
        labels = None
        avg_logits = None
        per_model = []
        for ckpt, backbone in zip(ckpts, backbones):
            image_size = input_size_for(backbone)
            if tta:
                dataset = TTADataset(data_dir, split, image_size)
            else:
                dataset = FolderDataset(data_dir, split, build_eval_transform(image_size))
            if labels is None:
                labels = np.array([y for _, y in dataset.samples])
                print(f'{split}: {len(dataset)} images | tta={tta} | device={device}')
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0 if tta else args.num_workers)

            model, size = load_model(ckpt, backbone, device)
            logits = predict_logits(model, loader, device, tta, image_size)
            preds = logits.argmax(axis=1)
            acc = 100.0 * accuracy_score(labels, preds)
            f1 = 100.0 * f1_score(labels, preds, average='macro')
            per_model.append({'backbone': backbone, 'acc': acc, 'macro_f1': f1})
            avg_logits = logits if avg_logits is None else avg_logits + logits
            del model
        return labels, avg_logits / len(ckpts), per_model

    if args.calibrate_on != 'none' and (args.calibrate_on != args.split or args.tta):
        val_labels, val_logits, _ = run_split(args.calibrate_on, args.tta)
        temp = fit_temperature(val_logits, val_labels)
        print(f'Calibrated on {args.calibrate_on}: T={temp:.3f}')
    else:
        temp = args.temperature if args.temperature else 1.0

    labels, avg_logits, per_model_acc = run_split(args.split, args.tta)
    avg_probs = _softmax(avg_logits / temp)

    preds = avg_probs.argmax(axis=1)
    acc = 100.0 * accuracy_score(labels, preds)
    macro_f1 = 100.0 * f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds, target_names=EMOTIONS, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()
    ece = compute_ece(avg_probs, labels)['ece']

    result = {
        'split': args.split,
        'tta': args.tta,
        'temperature': temp,
        'calibrate_on': None if args.calibrate_on == 'none' else args.calibrate_on,
        'ensemble_acc': acc,
        'ensemble_macro_f1': macro_f1,
        'ece': ece,
        'per_model': per_model_acc,
        'per_class_f1': {EMOTIONS[i]: report[EMOTIONS[i]]['f1-score'] for i in range(len(EMOTIONS))},
        'confusion_matrix': cm,
        'n_images': int(len(labels)),
        'checkpoints': ckpts,
        'backbones': backbones,
    }

    print(f'\nENSEMBLE {args.split}: acc={acc:.2f}% Macro-F1={macro_f1:.2f}% ECE={ece:.4f}')
    for m in per_model_acc:
        print(f"  {m['backbone']}: acc={m['acc']:.2f}% F1={m['macro_f1']:.2f}%")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f'wrote {out}')


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    return x - np.log(np.exp(x - x.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))


if __name__ == '__main__':
    main()
