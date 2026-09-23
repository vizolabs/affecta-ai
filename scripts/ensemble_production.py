"""Production ensemble leaderboard: DAN + DDAMFN++ + ViT with TTA + calibration.

Evaluated on the official RAF-DB test split (n=3068). Per-model data roots:
- ddamfn   : RetinaFace-realigned test_rf (matches its training recipe)
- others   : standard aligned test

Emits a JSON leaderboard with single-model rows, TTA rows, uniform/weighted
ensemble rows, and temperature-scaled (val-fitted) ECE for the best entry.

Usage:
    python scripts/ensemble_production.py                     # full sweep
    python scripts/ensemble_production.py --models dan ddamfn # subset
    python scripts/ensemble_production.py --no-tta            # fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import EMOTIONS, PROJECT_ROOT
from ml.models.production import load_production, build_transform, INPUT_SIZES
from ml.data.augmentation import ten_crop_flip_views
from ml.benchmarks.calibration import compute_ece, fit_temperature

DATA = PROJECT_ROOT / 'data/processed/rafdb'
OUT_DIR = PROJECT_ROOT / 'experiments/EXP-010-ENSEMBLE'

MODEL_ROOTS = {  # (test, val)
    'ddamfn': (DATA / 'test_rf', DATA / 'val_rf'),
    'dan': (DATA / 'test', DATA / 'val'),
    'vit_small': (DATA / 'test', DATA / 'val'),
    'efficientnet_b3': (DATA / 'test', DATA / 'val'),
}
BATCH = {'ddamfn': 256, 'dan': 128, 'vit_small': 128, 'efficientnet_b3': 64}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _loader(root: Path, transform, batch: int) -> DataLoader:
    ds = datasets.ImageFolder(str(root), transform=transform)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)


def collect_logits(model, root: Path, size: int, batch: int, tta: bool) -> tuple:
    """Returns (logits [N,7] in EMOTIONS order, targets [N]) as float32/int64."""
    if tta:
        tf = ten_crop_flip_views(size)
        loader = _loader(root, tf, batch)
    else:
        loader = _loader(root, build_transform(model.name), batch)
    logits, targets = [], []
    n_batches = len(loader)
    t0 = time.time()
    with torch.inference_mode():
        for i, (imgs, tgts) in enumerate(loader):
            if tta:
                b, v, c, h, w = imgs.shape
                out = model(imgs.view(b * v, c, h, w).to(model.gather.device if hasattr(model, 'gather') else 'cpu'))
                out = out.view(b, v, -1).mean(dim=1)
            else:
                dev = next(model.parameters()).device
                out = model(imgs.to(dev))
            logits.append(out.float().cpu().numpy())
            targets.append(tgts.numpy())
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f'    [{model.name}{"+tta" if tta else ""}] '
                      f'{i+1}/{n_batches} batches  {time.time()-t0:.0f}s', flush=True)
    return np.concatenate(logits), np.concatenate(targets)


def row(name: str, logits: np.ndarray, targets: np.ndarray) -> dict:
    probs = _softmax(logits)
    preds = probs.argmax(1)
    acc = float((preds == targets).mean() * 100)
    from sklearn.metrics import balanced_accuracy_score
    bacc = float(balanced_accuracy_score(targets, preds) * 100)
    ece = compute_ece(logits, targets)['ece']
    return {'name': name, 'test_acc': round(acc, 2), 'test_bacc': round(bacc, 2),
            'ece': round(float(ece), 4), 'n': int(len(targets))}


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['dan', 'ddamfn', 'vit_small'])
    ap.add_argument('--no-tta', action='store_true')
    ap.add_argument('--out', default=str(OUT_DIR / 'production_ensemble_results.json'))
    args = ap.parse_args()

    device = pick_device()
    print(f'device={device}  models={args.models}  tta={not args.no_tta}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets_ref = None
    test_logits, val_logits = {}, {}

    for name in args.models:
        print(f'\n== {name} ==', flush=True)
        model = load_production(name, device=device)
        test_root, val_root = MODEL_ROOTS[name]
        if not test_root.exists():
            print(f'  skip: missing {test_root}')
            continue
        logits, targets = collect_logits(model, test_root, model.input_size,
                                         BATCH[name], tta=False)
        test_logits[name] = logits
        if targets_ref is None:
            targets_ref = targets
        assert np.array_equal(targets, targets_ref), f'target order mismatch in {name}'
        if not args.no_tta:
            logits_t, _ = collect_logits(model, test_root, model.input_size,
                                         max(8, BATCH[name] // 4), tta=True)
            test_logits[f'{name}+tta'] = logits_t
        if val_root.exists():
            v_logits, _ = collect_logits(model, val_root, model.input_size,
                                         BATCH[name], tta=False)
            val_logits[name] = v_logits
        del model

    targets = targets_ref
    rows = []
    print('\n== single models ==')
    for key, lg in test_logits.items():
        r = row(key, lg, targets)
        rows.append(r)
        print(f"  {r['name']:18s} acc={r['test_acc']:6.2f}  bacc={r['test_bacc']:6.2f}  ece={r['ece']:.4f}")

    # ---- ensembles (uniform prob averages) ----
    def combo(label: str, keys: list):
        if not all(k in test_logits for k in keys):
            return None
        probs = np.mean([_softmax(test_logits[k]) for k in keys], axis=0)
        lg = np.log(probs + 1e-12)
        r = row(label, lg, targets)
        rows.append(r)
        print(f"  {r['name']:18s} acc={r['test_acc']:6.2f}  bacc={r['test_bacc']:6.2f}  ece={r['ece']:.4f}")
        return r

    print('\n== ensembles ==')
    combo('ens_dan+ddamfn', ['dan', 'ddamfn'])
    combo('ens_dan+vit', ['dan', 'vit_small'])
    combo('ens_dan+ddamfn+vit', ['dan', 'ddamfn', 'vit_small'])
    combo('ens_tta_dan+ddamfn', ['dan+tta', 'ddamfn+tta'])
    combo('ens_tta_dan+ddamfn+vit', ['dan+tta', 'ddamfn+tta', 'vit_small+tta'])
    combo('ens_best3_tta', ['dan+tta', 'ddamfn+tta', 'vit_small+tta'])

    # weighted: DAN-heavy (best single model dominates)
    if all(k in test_logits for k in ('dan+tta', 'ddamfn+tta', 'vit_small+tta')):
        w = np.array([0.5, 0.3, 0.2])
        probs = (w[0] * _softmax(test_logits['dan+tta'])
                 + w[1] * _softmax(test_logits['ddamfn+tta'])
                 + w[2] * _softmax(test_logits['vit_small+tta']))
        r = row('ens_w532_tta', np.log(probs + 1e-12), targets)
        rows.append(r)
        print(f"  {r['name']:18s} acc={r['test_acc']:6.2f}  bacc={r['test_bacc']:6.2f}  ece={r['ece']:.4f}")

    # ---- temperature scaling (val-fitted) on the best row's underlying probs ----
    best = max(rows, key=lambda r: r['test_acc'])
    print(f'\n== calibration for {best["name"]} ==')
    cal = {'target': best['name'], 'raw_ece': best['ece']}
    key = best['name']
    if key in test_logits and key in val_logits:
        T = fit_temperature(val_logits[key], _val_targets(val_root_for(key), val_logits[key]))
        cal['temperature'] = round(T, 4)
        cal['cal_ece'] = round(compute_ece(test_logits[key] / T, targets)['ece'], 4)
    elif key.startswith('ens'):
        # fit T on mean val probs if all members have val logits
        members = {'ens_tta_dan+ddamfn': ['dan', 'ddamfn'],
                   'ens_tta_dan+ddamfn+vit': ['dan', 'ddamfn', 'vit_small'],
                   'ens_best3_tta': ['dan', 'ddamfn', 'vit_small'],
                   'ens_w532_tta': ['dan', 'ddamfn', 'vit_small'],
                   'ens_dan+ddamfn+vit': ['dan', 'ddamfn', 'vit_small'],
                   'ens_dan+ddamfn': ['dan', 'ddamfn'],
                   'ens_dan+vit': ['dan', 'vit_small']}.get(key)
        if members and all(m in val_logits for m in members):
            vprobs = np.mean([_softmax(val_logits[m]) for m in members], axis=0)
            vlg = np.log(vprobs + 1e-12)
            vtargets = _val_targets(DATA / ('val_rf' if 'ddamfn' in members else 'val'),
                                    vlg) if False else _load_targets(DATA / 'val')
            T = fit_temperature(vlg, vtargets)
            probs_test = _softmax(np.array([r for r in [None]]) * 0) if False else None
            # apply T to ensemble test probs
            best_row_probs = None
            cal['temperature'] = round(T, 4)
            # recompute ensemble test probs for ECE
            if key == 'ens_tta_dan+ddamfn':
                p = np.mean([_softmax(test_logits[k]) for k in ('dan+tta', 'ddamfn+tta')], axis=0)
            elif key == 'ens_w532_tta':
                p = (0.5 * _softmax(test_logits['dan+tta'])
                     + 0.3 * _softmax(test_logits['ddamfn+tta'])
                     + 0.2 * _softmax(test_logits['vit_small+tta']))
            else:
                p = np.mean([_softmax(test_logits[k]) for k in
                             ('dan+tta', 'ddamfn+tta', 'vit_small+tta')], axis=0)
            lg = np.log(p + 1e-12) / T
            cal['cal_ece'] = round(compute_ece(lg, targets)['ece'], 4)
    rows.append({'name': f"cal::{best['name']}", **{k: v for k, v in cal.items()
                                                    if k in ('raw_ece', 'cal_ece', 'temperature')}})
    print(f"  raw_ece={cal.get('raw_ece')}  T={cal.get('temperature')}  cal_ece={cal.get('cal_ece')}")

    result = {
        'split': 'official RAF-DB test (n=3068)',
        'class_order': EMOTIONS,
        'device': device,
        'calibration': cal,
        'leaderboard': rows,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nsaved {args.out}')
    best_single = max((r for r in rows if '+' not in r['name'] and not r['name'].startswith('ens')
                       and not r['name'].startswith('cal')), key=lambda r: r['test_acc'])
    best_any = max((r for r in rows if not r['name'].startswith('cal')),
                   key=lambda r: r['test_acc'])
    print(f"BEST single: {best_single['name']} {best_single['test_acc']}%")
    print(f"BEST overall: {best_any['name']} {best_any['test_acc']}%")
    return 0


def _load_targets(val_root: Path) -> np.ndarray:
    ds = datasets.ImageFolder(str(val_root))
    return np.array(ds.targets, dtype=np.int64)


def _val_targets(root: Path, logits: np.ndarray) -> np.ndarray:
    return _load_targets(root)


def _val_root_for(key: str) -> Path:
    return MODEL_ROOTS.get(key, (None, DATA / 'val'))[1]


if __name__ == '__main__':
    raise SystemExit(main())
