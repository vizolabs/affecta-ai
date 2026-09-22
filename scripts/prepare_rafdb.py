"""Prepare RAF-DB basic (7-class) into the AFFECTA processed layout.

Two source layouts are supported:

1) Official raw layout (whdeng.cn/RAF/model1.html, "basic"):
       <raw>/basic/EmoLabel/list_patition_label.txt
       <raw>/basic/Image/aligned/*.jpg
   List file format: <image_name>\t<train|test>\t<label 1-7>

2) HuggingFace parquet mirrors (official train/test partition):
       <raw>/train-00000-of-00001.parquet
       <raw>/test-00000-of-00001.parquet
   Columns: image={bytes, path}, label in 1..7
   Split names are taken from the parquet (official partition preserved).

Label mapping (both sources):
    1=Surprise -> surprise, 2=Fear -> fear, 3=Disgust -> disgust,
    4=Happiness -> happy, 5=Sadness -> sad, 6=Anger -> angry, 7=Neutral -> neutral

Output layout (matches ml/data/dataset.py):
    <out>/{train,val,test}/<emotion>/*.jpg

Usage:
    python scripts/prepare_rafdb.py --raw data/raw/rafdb --out data/processed/rafdb
    python scripts/prepare_rafdb.py --raw data/raw/rafdb --out data/processed/rafdb \
        --source parquet --val-fraction 0.15
"""

import argparse
import json
import shutil
from pathlib import Path

from src.core.config import EMOTIONS

RAF_TO_EMOTION = {
    1: 'surprise',
    2: 'fear',
    3: 'disgust',
    4: 'happy',
    5: 'sad',
    6: 'angry',
    7: 'neutral',
}


def find_label_file(raw: Path) -> Path:
    candidates = []
    for p in raw.rglob('*.txt'):
        if 'label' in p.name.lower() or 'partition' in p.name.lower():
            candidates.append(p)
    return candidates[0] if candidates else None


def find_aligned_dir(raw: Path) -> Path:
    for p in raw.rglob('*'):
        if p.is_dir() and p.name.lower() in ('aligned', 'aligned_labeled', 'aligned_cropped'):
            return p
    for p in raw.rglob('*'):
        if p.is_dir() and p.name.lower() == 'image':
            return p
    return None


def parse_labels(label_file: Path) -> list:
    entries = []
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            image, partition, label = parts
        elif len(parts) == 2:
            image, label = parts
            partition = None
        else:
            continue
        entries.append((image, partition, int(label)))
    return entries


def detect_source(raw: Path) -> str:
    if list(raw.glob('*.parquet')):
        return 'parquet'
    if (raw / 'DATASET' / 'train').is_dir() and (raw / 'DATASET' / 'test').is_dir():
        return 'kaggle'
    if find_label_file(raw) is not None and find_aligned_dir(raw) is not None:
        return 'official'
    return 'none'


def _stratified_val_split(items: list, val_fraction: float) -> tuple:
    """Split [(image, emotion), ...] into (val_items, train_items), stratified."""
    import collections

    n = len(items)
    val_count = int(n * val_fraction)
    counts = collections.Counter(e for _, e in items)
    per_class = {}
    for emotion in EMOTIONS:
        per_class[emotion] = max(1, val_count * counts.get(emotion, 0) // max(1, n))

    val_entries = []
    remaining = []
    seen = collections.Counter()
    for image, emotion in items:
        if seen[emotion] < per_class.get(emotion, 0):
            val_entries.append((image, emotion))
            seen[emotion] += 1
        else:
            remaining.append((image, emotion))
    return val_entries, remaining


def prepare_official(raw: Path, out: Path, val_fraction: float) -> dict:
    label_file = find_label_file(raw)
    aligned_dir = find_aligned_dir(raw)
    if label_file is None or aligned_dir is None:
        raise SystemExit(
            f'RAF-DB official raw data not found under {raw} '
            '(label file or aligned image dir missing).'
        )
    print(f'label file: {label_file}')
    print(f'aligned dir: {aligned_dir}')
    entries = parse_labels(label_file)
    print(f'parsed {len(entries)} label entries')

    splits = {'train': [], 'test': []}
    for image, partition, label in entries:
        emotion = RAF_TO_EMOTION.get(label)
        if emotion is None:
            continue
        split = 'test' if partition == 'test' else 'train'
        splits[split].append((image, emotion))

    val_entries, train_entries = _stratified_val_split(splits['train'], val_fraction)
    splits['val'] = val_entries
    splits['train'] = train_entries

    for split, items in splits.items():
        done = 0
        for image, emotion in items:
            src = aligned_dir / image
            if not src.exists():
                continue
            dst_dir = out / split / emotion
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / image
            if not dst.exists():
                shutil.copy2(src, dst)
            done += 1
        print(f'[{split:5s}] {done}/{len(items)} images copied')

    return {'splits': {s: len(i) for s, i in splits.items()}}


def prepare_parquet(raw: Path, out: Path, val_fraction: float) -> dict:
    """Extract HF parquet mirrors into the processed folder layout.

    Official train/test partition is preserved; a stratified val holdout is
    carved from the train partition only (test stays untouched).
    """
    import pandas as pd

    train_files = sorted(raw.glob('train*.parquet'))
    test_files = sorted(raw.glob('test*.parquet'))
    if not train_files or not test_files:
        raise SystemExit(f'Expected train*.parquet and test*.parquet under {raw}')

    def _rows(files):
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df

    train_df = _rows(train_files)
    test_df = _rows(test_files)
    print(f'parquet: train={len(train_df)} test={len(test_df)}')

    def _collect(df) -> list:
        items = []
        for _, row in df.iterrows():
            emotion = RAF_TO_EMOTION.get(int(row['label']))
            if emotion is None:
                continue
            img = row['image']
            if isinstance(img, dict):
                data = img.get('bytes')
                name = img.get('path') or f'{row.name}.jpg'
            else:
                data = img
                name = f'{row.name}.jpg'
            items.append((name, emotion, data))
        return items

    train_items = _collect(train_df)  # (name, emotion, bytes)
    test_items = _collect(test_df)

    # Stratified val from train partition (keep bytes alongside)
    train_pairs = [(n, e) for n, e, _ in train_items]
    val_pairs, keep_pairs = _stratified_val_split(train_pairs, val_fraction)
    val_set = set(val_pairs)
    keep_set = set(keep_pairs)

    def _write(items, split_name, filter_set=None):
        done = 0
        for name, emotion, data in items:
            key = (name, emotion)
            if filter_set is not None and key not in filter_set:
                continue
            if data is None:
                continue
            dst_dir = out / split_name / emotion
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / name
            if not dst.exists():
                dst.write_bytes(data)
            done += 1
        print(f'[{split_name:5s}] {done} images written')
        return done

    n_train = _write(train_items, 'train', keep_set)
    n_val = _write(train_items, 'val', val_set)
    n_test = _write(test_items, 'test')

    return {
        'source': 'parquet',
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'official_train_total': len(train_items),
        'official_test_total': len(test_items),
    }


def prepare_kaggle(raw: Path, out: Path, val_fraction: float) -> dict:
    """Prepare the common Kaggle RAF-DB layout.

    Layout (DATASET/train/<label 1-7>/...jpg, DATASET/test/<label 1-7>/...jpg):
        <raw>/DATASET/train/1/train_00001_aligned.jpg
        <raw>/DATASET/test/1/test_00002_aligned.jpg
    Label comes from the parent directory name (1-7 -> emotion).
    Official train/test partition preserved; stratified val carved from train.
    """
    base = raw / 'DATASET'

    def _collect(split: str) -> list:
        items = []
        split_dir = base / split
        if not split_dir.is_dir():
            return items
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label = int(label_dir.name)
            except ValueError:
                continue
            emotion = RAF_TO_EMOTION.get(label)
            if emotion is None:
                continue
            for img in sorted(label_dir.glob('*.jpg')):
                items.append((img.name, emotion, img))
        return items

    train_items = _collect('train')
    test_items = _collect('test')
    print(f'kaggle: train={len(train_items)} test={len(test_items)}')

    train_pairs = [(n, e) for n, e, _ in train_items]
    val_pairs, keep_pairs = _stratified_val_split(train_pairs, val_fraction)
    val_set = set(val_pairs)
    keep_set = set(keep_pairs)

    def _write(items, split_name, filter_set=None):
        done = 0
        for name, emotion, src in items:
            key = (name, emotion)
            if filter_set is not None and key not in filter_set:
                continue
            dst_dir = out / split_name / emotion
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)
            done += 1
        print(f'[{split_name:5s}] {done} images written')
        return done

    n_train = _write(train_items, 'train', keep_set)
    n_val = _write(train_items, 'val', val_set)
    n_test = _write(test_items, 'test')

    return {
        'source': 'kaggle',
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'official_train_total': len(train_items),
        'official_test_total': len(test_items),
    }


def write_summary(out: Path, summary: dict):
    summaries = {}
    for split in ('train', 'val', 'test'):
        summaries[split] = {
            e: len(list((out / split / e).glob('*'))) if (out / split / e).exists() else 0
            for e in EMOTIONS
        }
        print(f'{split}: ' + ', '.join(f'{e}={summaries[split][e]}' for e in EMOTIONS))
    summary['per_class'] = summaries
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'wrote {out / "summary.json"}')


def main():
    parser = argparse.ArgumentParser(description='Prepare RAF-DB into processed layout')
    parser.add_argument('--raw', type=Path, default=Path('data/raw/rafdb'))
    parser.add_argument('--out', type=Path, default=Path('data/processed/rafdb'))
    parser.add_argument('--val-fraction', type=float, default=0.15,
                        help='Stratified val holdout from the train partition')
    parser.add_argument('--source', choices=['auto', 'official', 'parquet', 'kaggle'], default='auto')
    args = parser.parse_args()

    source = args.source
    if source == 'auto':
        source = detect_source(args.raw)
        if source == 'none':
            raise SystemExit(
                f'No RAF-DB source found under {args.raw} '
                '(need parquet files or official label+image layout).'
            )
    print(f'source: {source}')

    args.out.mkdir(parents=True, exist_ok=True)
    if source == 'parquet':
        summary = prepare_parquet(args.raw, args.out, args.val_fraction)
    elif source == 'kaggle':
        summary = prepare_kaggle(args.raw, args.out, args.val_fraction)
    else:
        summary = prepare_official(args.raw, args.out, args.val_fraction)
        summary['source'] = 'official'
    write_summary(args.out, summary)


if __name__ == '__main__':
    main()