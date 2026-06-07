#!/usr/bin/env python3
"""
Full-ranking evaluation for sequential recommendation.
Ranks test item against ALL catalog items (minus training items).
"""

import os
import sys
import math
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

BASE        = Path(__file__).parent
DATA_DIR    = BASE / 'data'
RESULTS_DIR = BASE / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE))
from dataset import load_data, EvalDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(BASE / 'run.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def evaluate_model(model, train_seqs, eval_items, num_items,
                   device, max_len=50, batch_size=256):
    """Batched full-ranking evaluation. Pre-computes all item embeddings."""
    model.eval()
    dataset = EvalDataset(train_seqs, eval_items, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ks = [1, 5, 10]
    recalls = {k: 0.0 for k in ks}
    ndcgs   = {k: 0.0 for k in ks}
    total   = 0

    with torch.no_grad():
        all_item_ids  = torch.arange(1, num_items + 1, device=device)
        all_item_embs = model.item_emb(all_item_ids)  # (N, D)

        for uid_batch, seq, pos_item in loader:
            seq    = seq.to(device)
            B      = seq.shape[0]
            h_last = model.encode(seq)  # (B, D)

            all_scores = (h_last @ all_item_embs.T).cpu().numpy()  # (B, N)

            for b in range(B):
                uid    = uid_batch[b].item()
                pos    = pos_item[b].item()
                seen   = train_seqs.get(uid, [])
                scores = all_scores[b].copy()

                for item in seen:
                    if 1 <= item <= num_items:
                        scores[item - 1] = -1e9

                pos_score = scores[pos - 1]
                rank = int((scores > pos_score).sum())

                for k in ks:
                    if rank < k:
                        recalls[k] += 1.0
                        ndcgs[k]   += 1.0 / math.log2(rank + 2)
                total += 1

    metrics = {}
    for k in ks:
        metrics[f'Recall@{k}'] = recalls[k] / total
        metrics[f'NDCG@{k}']   = ndcgs[k]   / total

    return metrics, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True, choices=['sasrec', 'gru4rec'])
    parser.add_argument('--dataset',    required=True, choices=['beauty', 'scientific', 'sports', 'game', 'instrument'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--max_len',    type=int, default=50)
    parser.add_argument('--split',      default='test', choices=['valid', 'test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"evaluating {args.model} on {args.dataset} ({args.split}) -- device={device}")

    train_seqs, valid_items, test_items, stats = load_data(str(DATA_DIR), args.dataset)
    num_items  = int(stats['num_items'])
    eval_items = test_items if args.split == 'test' else valid_items

    ckpt        = torch.load(args.checkpoint, map_location=device)
    model_state = ckpt['model_state']
    model_args  = ckpt['model_args']

    if args.model == 'sasrec':
        from models.sasrec import SASRec
        model = SASRec(**model_args)
    else:
        from models.gru4rec import GRU4Rec
        model = GRU4Rec(**model_args)

    model.load_state_dict(model_state)
    model.to(device)

    metrics, total = evaluate_model(
        model, train_seqs, eval_items, num_items, device, args.max_len
    )

    log.info(f"results ({total} users):")
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}")

    out_path = RESULTS_DIR / f"{args.model}_{args.dataset}_results.txt"
    with open(out_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Num test users: {total}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    log.info(f"saved to {out_path}")


if __name__ == '__main__':
    main()
