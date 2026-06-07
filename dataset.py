"""
Dataset utilities for sequential recommendation.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


def load_data(data_dir, dataset):
    base = os.path.join(data_dir, dataset)

    train_seqs = {}
    with open(os.path.join(base, 'train.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                uid = int(parts[0])
                items = list(map(int, parts[1:]))
                train_seqs[uid] = items

    valid_items = {}
    with open(os.path.join(base, 'valid.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                valid_items[int(parts[0])] = int(parts[1])

    test_items = {}
    with open(os.path.join(base, 'test.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                test_items[int(parts[0])] = int(parts[1])

    stats = {}
    with open(os.path.join(base, 'stats.txt')) as f:
        for line in f:
            k, v = line.strip().split()
            stats[k] = float(v)

    return train_seqs, valid_items, test_items, stats


def pad_seq(seq, max_len):
    """Right-pad or truncate sequence to max_len."""
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    return [0] * pad_len + seq


class TrainDataset(Dataset):
    """
    For each user, generate (seq, pos, neg) samples.
    seq: items seen before pos (up to max_len)
    pos: a positive item
    neg: a randomly sampled negative item
    """
    def __init__(self, train_seqs, num_items, max_len=50):
        self.max_len   = max_len
        self.num_items = num_items
        self.samples   = []

        for uid, seq in train_seqs.items():
            if len(seq) < 1:
                continue
            for i in range(1, len(seq) + 1):
                pos = seq[i - 1]
                context = seq[:i - 1]
                self.samples.append((context, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, pos = self.samples[idx]
        padded = pad_seq(context, self.max_len)

        # Sample negative
        neg = np.random.randint(1, self.num_items + 1)
        while neg == pos:
            neg = np.random.randint(1, self.num_items + 1)

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(pos,    dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
        )


class EvalDataset(Dataset):
    """
    For evaluation: returns (uid, seq, pos_item) per user.
    train_set must be looked up externally using uid.
    """
    def __init__(self, train_seqs, eval_items, max_len=50):
        self.max_len = max_len
        self.samples = []
        for uid, pos in eval_items.items():
            seq = train_seqs.get(uid, [])
            padded = pad_seq(seq, max_len)
            self.samples.append((uid, padded, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, padded, pos = self.samples[idx]
        return (
            uid,
            torch.tensor(padded, dtype=torch.long),
            pos,
        )
