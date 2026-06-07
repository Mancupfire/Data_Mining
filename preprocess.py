#!/usr/bin/env python3
"""
Preprocessing pipeline for Amazon Review datasets.
Performs 5-core filtering, leave-one-out split, and re-indexing.
"""

import os
import json
import gzip
import argparse
import logging
from collections import defaultdict
from pathlib import Path

BASE    = Path(__file__).parent
RAW_DIR = BASE / 'data' / 'raw'
DATA_DIR = BASE / 'data'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(BASE / 'run.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

DATASET_FILES = {
    'beauty':     'All_Beauty.jsonl.gz',
    'sports':     'Sports_and_Outdoors.jsonl.gz',
    'scientific': 'Industrial_and_Scientific.jsonl.gz',
}


def load_interactions(filepath):
    interactions = []
    log.info(f"loading {filepath}")
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Amazon 2023: user_id / parent_asin / timestamp
            # Amazon 5-core (old): reviewerID / asin / unixReviewTime
            user = obj.get('user_id') or obj.get('reviewerID')
            item = obj.get('parent_asin') or obj.get('asin')
            ts   = obj.get('timestamp') or obj.get('unixReviewTime') or 0
            if user and item:
                try:
                    interactions.append((str(user), str(item), int(float(ts))))
                except (ValueError, TypeError):
                    interactions.append((str(user), str(item), 0))
    log.info(f"  loaded {len(interactions):,} raw interactions")
    return interactions


def five_core_filter(interactions):
    while True:
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for u, i, _ in interactions:
            user_counts[u] += 1
            item_counts[i] += 1

        before = len(interactions)
        interactions = [(u, i, t) for u, i, t in interactions
                        if user_counts[u] >= 5 and item_counts[i] >= 5]
        after = len(interactions)
        log.info(f"  5-core pass: {before:,} -> {after:,}")
        if before == after:
            break
    return interactions


def process_dataset(name):
    fname = DATASET_FILES[name]
    raw_path = RAW_DIR / fname
    if not raw_path.exists():
        log.error(f"raw file not found: {raw_path}")
        return False

    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== processing {name} ===")

    interactions = load_interactions(raw_path)
    interactions = five_core_filter(interactions)

    user_items = defaultdict(list)
    for u, i, t in interactions:
        user_items[u].append((t, i))
    for u in user_items:
        user_items[u].sort()

    user_items = {u: v for u, v in user_items.items() if len(v) >= 3}

    remaining_items = set()
    for seqs in user_items.values():
        for _, i in seqs:
            remaining_items.add(i)

    user2id = {u: idx+1 for idx, u in enumerate(sorted(user_items))}
    item2id = {i: idx+1 for idx, i in enumerate(sorted(remaining_items))}

    num_users = len(user2id)
    num_items = len(item2id)
    num_interactions = sum(len(v) for v in user_items.values())
    sparsity = 1 - num_interactions / (num_users * num_items)

    log.info(f"  users={num_users:,}, items={num_items:,}, "
             f"interactions={num_interactions:,}, sparsity={sparsity:.6f}")

    train_lines, valid_lines, test_lines = [], [], []
    for u_raw, seqs in user_items.items():
        uid = user2id[u_raw]
        items = [item2id[i] for _, i in seqs]
        test_item  = items[-1]
        valid_item = items[-2]
        train_seq  = items[:-2]

        train_lines.append(str(uid) + ' ' + ' '.join(map(str, train_seq)))
        valid_lines.append(f"{uid} {valid_item}")
        test_lines.append(f"{uid} {test_item}")

    def write_lines(path, lines):
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    write_lines(out_dir / 'train.txt', train_lines)
    write_lines(out_dir / 'valid.txt', valid_lines)
    write_lines(out_dir / 'test.txt',  test_lines)

    with open(out_dir / 'stats.txt', 'w') as f:
        f.write(f"num_users {num_users}\n")
        f.write(f"num_items {num_items}\n")
        f.write(f"num_interactions {num_interactions}\n")
        f.write(f"sparsity {sparsity:.6f}\n")

    log.info(f"  saved to {out_dir}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['beauty', 'scientific', 'sports'])
    args = parser.parse_args()

    for ds in args.datasets:
        process_dataset(ds)
