"""Supervised relauncher for unattended training runs.

Runs ml/training/train_expression.py and, on any unexpected exit or
termination, relaunches it with --resume pointing at the latest best.pt, so
long training runs survive machine sleep / process-group kills without losing
progress. Halts on clean completion (exit 0) or user interrupt (exit 130).

Usage (standard detach pattern):
    PYTHONPATH="." nohup python3 -c "import os,sys; os.setsid(); os.execvp(sys.executable, \
        [sys.executable, 'scripts/supervised_train.py', '--experiment-id', 'EXP-XXX', \
         '--backbone', 'vgg16', '--batch-size', '32'])" > experiments/EXP-XXX/train_resume.log 2>&1 &
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def build_launch_args(base_args: list, resume_ckpt: Path) -> list:
    cmd = [sys.executable, 'ml/training/train_expression.py'] + base_args
    if resume_ckpt.exists():
        cmd += ['--resume', str(resume_ckpt)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Supervised training relauncher')
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-restarts', type=int, default=20)
    parser.add_argument('--sleep-between', type=float, default=30.0)
    # Optional Phase-0 passthrough flags (forwarded verbatim to train_expression.py)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--augmentation', choices=['baseline', 'strong'], default=None)
    parser.add_argument('--loss', choices=['ce', 'focal'], default=None)
    parser.add_argument('--mixup', type=float, default=None)
    parser.add_argument('--cutmix', type=float, default=None)
    parser.add_argument('--grad-accum', type=int, default=None)
    parser.add_argument('--amp', action='store_true', default=None)
    parser.add_argument('--swa-start', type=int, default=None)
    parser.add_argument('--init-weights', default=None)
    args, unknown = parser.parse_known_args()

    base_args = [
        '--backbone', args.backbone,
        '--experiment-id', args.experiment_id,
        '--batch-size', str(args.batch_size),
    ]
    optional = {
        '--data-dir': args.data_dir,
        '--epochs': args.epochs,
        '--augmentation': args.augmentation,
        '--loss': args.loss,
        '--mixup': args.mixup,
        '--cutmix': args.cutmix,
        '--grad-accum': args.grad_accum,
        '--swa-start': args.swa_start,
        '--init-weights': args.init_weights,
    }
    for flag, val in optional.items():
        if val is not None:
            base_args += [flag, str(val)]
    if args.amp:
        base_args.append('--amp')
    # Unknown args are forwarded as-is (e.g. custom experiment flags).
    base_args += unknown
    best_ckpt = PROJECT_ROOT / 'experiments' / args.experiment_id / 'checkpoints' / 'best.pt'

    restarts = 0
    while True:
        cmd = build_launch_args(base_args, best_ckpt)
        suffix = ' (fresh)' if not best_ckpt.exists() else f' (resume from {best_ckpt.name})'
        print(f'[supervisor] launching: {" ".join(cmd[-4:])}{suffix}', flush=True)
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)

        if proc.returncode == 0:
            print('[supervisor] training completed (exit 0); stopping.', flush=True)
            break
        if proc.returncode == 130:
            print('[supervisor] user interrupt (exit 130); stopping.', flush=True)
            break

        restarts += 1
        if restarts > args.max_restarts:
            print(f'[supervisor] exceeded {args.max_restarts} restarts; giving up.', flush=True)
            sys.exit(1)

        print(
            f'[supervisor] training exited rc={proc.returncode}; '
            f'relaunching in {args.sleep_between:.0f}s (restart {restarts}/{args.max_restarts})',
            flush=True,
        )
        time.sleep(args.sleep_between)


if __name__ == '__main__':
    main()