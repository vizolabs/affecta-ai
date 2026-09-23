"""Phase 4 XAI probe: per-class facial-region attribution study.

Question: for correctly-classified predictions, do the weak emotion classes
(fear/sad/angry) show a distinguishable, concentrated region-attribution pattern
vs. the strong classes (happy/surprise)?

Method: Grad x Input relevance (fast, robust) aggregated over disjoint
positional facial zones. FER2013 inputs are pre-aligned face crops, so zones are
computed as fractions of the input; this is an approximation until a landmark
detector is added for Phase 4 production.

Usage:
    python ml/benchmarks/xai_probe.py --checkpoint <best.pt> [--backbone vgg16] \
        --samples-per-class 30 [--device auto] [--output <json>]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.data.dataset import FERDataset
from ml.models.backbones import build_expression_model, input_size_for
from ml.xai.lrp import get_lrp_engine
from ml.xai.lrp_epsilon import get_epsilon_lrp
from src.core.config import EMOTIONS, DATA_DIR

ZONE_DEFS = {
    'brows': (0.20, 0.32, 0.0, 1.0),
    'eyes': (0.34, 0.48, 0.0, 1.0),
    'nose': (0.50, 0.60, 0.15, 0.85),
    'cheeks': (0.50, 0.64, 0.0, 0.20, 0.80, 1.0),
    'mouth': (0.66, 0.82, 0.10, 0.90),
}


def zone_mask(name: str, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    defs = ZONE_DEFS[name]
    if name == 'cheeks':
        y0, y1, x0l, x1l, x0r, x1r = defs
        mask[int(h * y0):int(h * y1), int(w * x0l):int(w * x1l)] = 1.0
        mask[int(h * y0):int(h * y1), int(w * x0r):int(w * x1r)] = 1.0
    else:
        y0, y1, x0, x1 = defs
        mask[int(h * y0):int(h * y1), int(w * x0):int(w * x1)] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser(description='Per-class region attribution probe')
    parser.add_argument('--checkpoint', default='experiments/EXP-008-VIT-SMALL/checkpoints/best.pt')
    parser.add_argument('--backbone', default='vgg16',
                        choices=['vgg16', 'resnet50', 'efficientnet_b2', 'efficientnet_b3',
                                 'vit_small_patch16_224', 'vit_base_patch16_224'])
    parser.add_argument('--data-dir', default=str(DATA_DIR / 'processed'))
    parser.add_argument('--samples-per-class', type=int, default=30)
    parser.add_argument('--method', default='grad', choices=['grad', 'lrp'])
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    device = torch.device(
        'mps' if args.device == 'auto' and torch.backends.mps.is_available()
        else 'cpu' if args.device == 'auto' else args.device
    )
    image_size = input_size_for(args.backbone)

    model = build_expression_model(backbone=args.backbone, pretrained=False)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval().to(device)

    val_ds = FERDataset(args.data_dir, split='val', image_size=image_size)
    by_label = {}
    for path, label in val_ds.samples:
        by_label.setdefault(label, []).append(path)

    eng = get_lrp_engine() if args.method == 'grad' else get_epsilon_lrp()
    zones = list(ZONE_DEFS.keys())
    rng = np.random.RandomState(0)
    results = {}
    correct_total = 0

    for cls_idx, emotion in enumerate(EMOTIONS):
        paths = by_label.get(cls_idx, [])
        rng.shuffle(paths)
        vecs = []
        used = 0
        for path in paths:
            if used >= args.samples_per_class:
                break
            img = Image.open(path).convert('RGB')
            tensor = val_ds.transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor).argmax(1).item()
            if pred != cls_idx:
                continue
            used += 1
            correct_total += 1
            rel = eng.compute(model, tensor, target_class=cls_idx)
            rel_np = rel.detach().cpu().squeeze().numpy()
            if rel_np.ndim == 3:
                rel_np = rel_np.sum(axis=0)
            h, w = rel_np.shape
            abs_map = np.abs(rel_np)
            totals = [float(abs_map[zone_mask(z, h, w).astype(bool)].sum()) for z in zones]
            s = sum(totals)
            vecs.append([t / s if s > 0 else 0.0 for t in totals])
        if vecs:
            arr = np.array(vecs)
            results[emotion] = {
                'n': len(vecs),
                'mean': {z: float(m) for z, m in zip(zones, arr.mean(axis=0))},
                'std': {z: float(sd) for z, sd in zip(zones, arr.std(axis=0))},
            }

    out_path = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / f'xai_probe_regions_{args.method}.json'
    out_path.write_text(json.dumps({'backbone': args.backbone, 'method': args.method, 'zones': zones, 'results': results}, indent=2))

    print(f'correct predictions used: {correct_total}\n')
    print(f'{"class":9s} {"n":>3s} ' + ''.join(f'{z:>9s}' for z in zones))
    for emotion in EMOTIONS:
        r = results.get(emotion)
        if not r:
            continue
        row = f'{emotion:9s} {r["n"]:3d} '
        row += ''.join(f'{r["mean"][z]:9.3f}' for z in zones)
        print(row)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()