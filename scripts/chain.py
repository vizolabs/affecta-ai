"""Automated experiment handoff chain.

Waits for the previous training experiment to finish cleanly (trainer exited
and official test_metrics.json written), then:
  1. runs the latency benchmark on the previous experiment's frozen best.pt,
  2. launches the next experiment's supervised training in its own session.

Intended to run detached (nohup + setsid). Halts with an error if the previous
experiment abandons without producing metrics.

Usage:
    python scripts/chain.py \
        --prev-exp EXP-006-EFFICIENTNET-B2 --prev-backbone efficientnet_b2 \
        --next-exp EXP-007-EFFICIENTNET-B3 --next-backbone efficientnet_b3 \
        --next-data-dir data/processed/rafdb --next-augmentation strong
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
POLL_SECONDS = 20
MAX_WAIT_SECONDS = 16 * 3600


def trainer_running(exp_id: str) -> bool:
    out = subprocess.run(
        ['pgrep', '-f', f'train_expression.py --backbone'],
        capture_output=True,
        text=True,
    )
    pids = [p for p in out.stdout.split() if p]
    return bool(pids)


def metrics_exist(exp_id: str) -> bool:
    return (PROJECT_ROOT / 'experiments' / exp_id / 'test' / 'test_metrics.json').exists()


def run_latency(exp_id: str, backbone: str) -> None:
    best = PROJECT_ROOT / 'experiments' / exp_id / 'checkpoints' / 'best.pt'
    out = PROJECT_ROOT / 'experiments' / exp_id / 'latency.json'
    if not best.exists():
        print(f'[chain] ERROR: no best.pt for {exp_id}; skipping latency', flush=True)
        return
    cmd = [
        sys.executable, 'ml/benchmarks/latency.py',
        '--checkpoint', str(best),
        '--backbone', backbone,
        '--iterations', '200', '--warmup', '30',
        '--output', str(out),
    ]
    print(f'[chain] latency bench: {" ".join(cmd)}', flush=True)
    r = subprocess.run(
        ['env', f'PYTHONPATH={PROJECT_ROOT}', sys.executable] + cmd[1:],
        cwd=PROJECT_ROOT,
    )
    if r.returncode != 0:
        print('[chain] WARNING: latency bench exited non-zero', flush=True)


def spawn_supervised(exp_id: str, backbone: str, batch_size: int, extra: list) -> int:
    log_path = PROJECT_ROOT / 'experiments' / exp_id / 'train_resume.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, 'w')
    cmd = [
        sys.executable, 'scripts/supervised_train.py',
        '--backbone', backbone,
        '--experiment-id', exp_id,
        '--batch-size', str(batch_size),
    ] + extra
    p = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return p.pid


def main():
    parser = argparse.ArgumentParser(description='Experiment handoff chain')
    parser.add_argument('--prev-exp', required=True)
    parser.add_argument('--prev-backbone', required=True)
    parser.add_argument('--next-exp', required=True)
    parser.add_argument('--next-backbone', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--next-data-dir', default=None, help='Forwarded to supervised_train --data-dir')
    parser.add_argument('--next-augmentation', choices=['baseline', 'strong'], default=None)
    parser.add_argument('--next-loss', choices=['ce', 'focal'], default=None)
    parser.add_argument('--next-mixup', type=float, default=None)
    parser.add_argument('--next-cutmix', type=float, default=None)
    parser.add_argument('--next-epochs', type=int, default=None)
    parser.add_argument('--next-swa-start', type=int, default=None)
    parser.add_argument('--next-extra', default='', help='Extra raw args space-separated')
    args = parser.parse_args()

    extra = []
    if args.next_data_dir:
        extra += ['--data-dir', args.next_data_dir]
    if args.next_augmentation:
        extra += ['--augmentation', args.next_augmentation]
    if args.next_loss:
        extra += ['--loss', args.next_loss]
    if args.next_mixup is not None:
        extra += ['--mixup', str(args.next_mixup)]
    if args.next_cutmix is not None:
        extra += ['--cutmix', str(args.next_cutmix)]
    if args.next_epochs is not None:
        extra += ['--epochs', str(args.next_epochs)]
    if args.next_swa_start is not None:
        extra += ['--swa-start', str(args.next_swa_start)]
    if args.next_extra:
        extra += args.next_extra.split()

    print(f'[chain] watching for {args.prev_exp} to finish...', flush=True)
    waited = 0
    while waited < MAX_WAIT_SECONDS:
        if metrics_exist(args.prev_exp) and not trainer_running(args.prev_exp):
            print(f'[chain] {args.prev_exp} finished with metrics.', flush=True)
            break
        if not trainer_running(args.prev_exp) and not metrics_exist(args.prev_exp):
            print(f'[chain] {args.prev_exp} ended WITHOUT metrics; aborting chain.', flush=True)
            sys.exit(1)
        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS
    else:
        print('[chain] timeout waiting for previous experiment; aborting.', flush=True)
        sys.exit(1)

    run_latency(args.prev_exp, args.prev_backbone)

    pid = spawn_supervised(args.next_exp, args.next_backbone, args.batch_size, extra)
    print(f'[chain] launched {args.next_exp} supervised (pid {pid}) extra={extra}.', flush=True)


if __name__ == '__main__':
    main()