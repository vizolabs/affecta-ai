"""Inference latency + memory benchmark.

Loads a frozen best.pt checkpoint (full-state format from
ml/training/checkpoint.py) and measures single-image forward latency and
resident-set memory on the current device (MPS on macOS, else CPU).

Usage:
    python ml/benchmarks/latency.py --checkpoint <best.pt> [--iterations 200] [--warmup 30] [--output <path>]
"""

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch

from ml.models.backbones import build_expression_model, input_size_for

DEFAULTS = {
    'vgg16': 224,
    'resnet50': 224,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'vit_small_patch16_224': 224,
    'vit_base_patch16_224': 224,
}


def resolve_device(preferred: str) -> torch.device:
    if preferred == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(preferred)


def measure(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int,
    batch_size: int,
    iterations: int,
    warmup: int,
) -> dict:
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    per_image = [t / batch_size for t in latencies]
    return {
        'latency_ms_per_image': {
            'mean': round(statistics.mean(per_image), 3),
            'median': round(statistics.median(per_image), 3),
            'p95': round(sorted(per_image)[int(len(per_image) * 0.95) - 1], 3),
            'std': round(statistics.stdev(per_image), 3) if len(per_image) > 1 else 0.0,
        },
        'iterations': iterations,
        'warmup': warmup,
        'batch_size': batch_size,
        'input_size': input_size,
        'device': str(device),
    }


def memory_snapshot() -> dict:
    stats = {}
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats['peak_rss_bytes'] = rss_bytes
    stats['peak_rss_mib'] = round(rss_bytes / (1024 * 1024), 2)
    if torch.cuda.is_available():
        try:
            stats['cuda_allocated_bytes'] = torch.cuda.memory_allocated()
            stats['cuda_allocated_mib'] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass
    if torch.backends.mps.is_available():
        try:
            stats['mps_allocated_bytes'] = torch.mps.current_allocated_memory()
            stats['mps_allocated_mib'] = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
        except Exception:
            pass
    return stats


def main():
    parser = argparse.ArgumentParser(description='Inference latency + memory benchmark')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to best.pt checkpoint')
    parser.add_argument('--backbone', default=None,
                        choices=['vgg16', 'resnet50', 'efficientnet_b2', 'efficientnet_b3',
                                 'efficientnet_b4', 'vit_small_patch16_224', 'vit_base_patch16_224'],
                        help='Backbone name (default: read from checkpoint config)')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='auto', help='auto | cuda | mps | cpu')
    parser.add_argument('--output', default=None, help='Output JSON path (default: next to checkpoint)')
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    backbone = args.backbone or (ckpt.get('config') or {}).get('backbone')
    if backbone is None:
        raise SystemExit('Could not determine backbone; pass --backbone explicitly.')
    input_size = DEFAULTS.get(backbone) or input_size_for(backbone)

    device = resolve_device(args.device)
    model = build_expression_model(backbone, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    results = measure(
        model, device, input_size, args.batch_size, args.iterations, args.warmup
    )
    results.update(memory_snapshot())
    results['checkpoint'] = str(ckpt_path)
    results['backbone'] = backbone
    results['checkpoint_epoch'] = ckpt.get('epoch')
    results['checkpoint_stage'] = ckpt.get('current_stage')
    results['params'] = sum(p.numel() for p in model.parameters())

    out_path = Path(args.output) if args.output else ckpt_path.parent / 'latency.json'
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f'\nWrote: {out_path}')


if __name__ == '__main__':
    main()