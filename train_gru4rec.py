#!/usr/bin/env python3
"""
Training script for GRU4Rec.
"""

import os
import sys
import math
import argparse
import logging
import subprocess
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

BASE     = Path(__file__).parent
DATA_DIR = BASE / 'data'
CKPT_DIR = BASE / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE))
from dataset import load_data, TrainDataset, EvalDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(BASE / 'run.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def ndcg_at_k(rank, k):
    return 1.0 / math.log2(rank + 2) if rank < k else 0.0


def evaluate_quick(model, train_seqs, eval_items, num_items,
                   device, max_len=50, num_neg=99):
    model.eval()
    dataset = EvalDataset(train_seqs, eval_items, max_len)

    hits10, ndcg10, total = 0.0, 0.0, 0

    with torch.no_grad():
        for uid_batch, seq, pos_item in DataLoader(dataset, batch_size=256,
                                                    shuffle=False, num_workers=2):
            seq    = seq.to(device)
            B      = seq.shape[0]
            h_last = model.encode(seq)

            for b in range(B):
                uid  = uid_batch[b].item()
                pos  = pos_item[b].item()
                seen = set(train_seqs.get(uid, []))

                negs = []
                while len(negs) < num_neg:
                    n = np.random.randint(1, num_items + 1)
                    if n != pos and n not in seen:
                        negs.append(n)

                cands  = torch.tensor([pos] + negs, dtype=torch.long, device=device)
                c_embs = model.item_emb(cands)
                scores = (c_embs @ h_last[b]).cpu().numpy()
                rank   = int((scores > scores[0]).sum())

                if rank < 10:
                    hits10 += 1
                    ndcg10 += ndcg_at_k(rank, 10)
                total += 1

    return hits10 / total, ndcg10 / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"training GRU4Rec on {args.dataset} (device={device})")

    train_seqs, valid_items, test_items, stats = load_data(str(DATA_DIR), args.dataset)
    num_items = int(stats['num_items'])
    log.info(f"  items={num_items}, users={int(stats['num_users'])}")

    from models.gru4rec import GRU4Rec
    model_args = dict(
        num_items=num_items,
        hidden_dim=100,
        max_len=args.max_len,
        num_layers=1,
        dropout=0.2,
    )
    model = GRU4Rec(**model_args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    train_ds = TrainDataset(train_seqs, num_items, args.max_len)
    loader   = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=(device.type == 'cuda'))

    best_ndcg = -1.0
    best_ckpt = CKPT_DIR / f'gru4rec_{args.dataset}_best.pt'

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for seq, pos, neg in loader:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_neg = torch.stack([pos, neg], dim=1)
            logits  = model.predict(seq, pos_neg)

            loss = bce(logits[:, 0], torch.ones(len(logits), device=device)) + \
                   bce(logits[:, 1], torch.zeros(len(logits), device=device))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        if epoch % 10 == 0:
            log.info(f"  epoch {epoch}/{args.epochs} loss={total_loss/max(steps,1):.4f}")

        if epoch % 20 == 0 or epoch == args.epochs:
            hr10, ndcg10 = evaluate_quick(
                model, train_seqs, valid_items, num_items, device, args.max_len
            )
            log.info(f"  [valid] HR@10={hr10:.4f} NDCG@10={ndcg10:.4f}")
            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'model_args': model_args,
                    'ndcg10': ndcg10,
                }, best_ckpt)
                log.info(f"  saved best checkpoint (NDCG@10={ndcg10:.4f})")

    log.info(f"training done. best NDCG@10={best_ndcg:.4f}")
    return best_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    required=True, choices=['beauty', 'scientific', 'sports', 'game', 'instrument'])
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--max_len',    type=int, default=50)
    args = parser.parse_args()

    best_ckpt = train(args)

    log.info("running full test evaluation...")
    subprocess.run(
        [sys.executable, str(BASE / 'evaluate.py'),
         '--model', 'gru4rec',
         '--dataset', args.dataset,
         '--checkpoint', str(best_ckpt),
         '--split', 'test'],
        cwd=BASE
    )


if __name__ == '__main__':
    main()
