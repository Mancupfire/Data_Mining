#!/usr/bin/env python3
"""
Generate RESULTS_SUMMARY.txt from per-model result files and stats.
"""

import os
from pathlib import Path

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / 'results'
DATA_DIR    = BASE / 'data'
OUT_PATH    = BASE / 'RESULTS_SUMMARY.txt'

MODELS   = ['sasrec', 'gru4rec']
DATASETS = ['game', 'instrument', 'scientific']
METRICS  = ['Recall@1', 'Recall@5', 'Recall@10', 'NDCG@1', 'NDCG@5', 'NDCG@10']


def load_results(model, dataset):
    path = RESULTS_DIR / f'{model}_{dataset}_results.txt'
    if not path.exists():
        return None
    r = {}
    with open(path) as f:
        for line in f:
            for m in METRICS:
                if line.startswith(m + ':'):
                    r[m] = float(line.split(':')[1].strip())
    return r


def load_stats(dataset):
    path = DATA_DIR / dataset / 'stats.txt'
    if not path.exists():
        return {}
    s = {}
    with open(path) as f:
        for line in f:
            k, v = line.strip().split()
            s[k] = v
    return s


lines = []
lines.append("=" * 80)
lines.append("results summary -- sequential recommendation baselines")
lines.append("=" * 80)
lines.append("")

lines.append("## dataset statistics")
lines.append(f"{'dataset':<15} {'#users':>10} {'#items':>10} {'#interactions':>15} {'sparsity':>12}")
lines.append("-" * 65)
for ds in DATASETS:
    s = load_stats(ds)
    if s:
        sp = s.get('sparsity')
        sp_str = f"{float(sp):.6f}" if sp is not None else 'N/A'
        lines.append(f"{ds:<15} {s.get('num_users','N/A'):>10} {s.get('num_items','N/A'):>10} "
                     f"{s.get('num_interactions','N/A'):>15} {sp_str:>12}")
    else:
        lines.append(f"{ds:<15} {'N/A':>10} {'N/A':>10} {'N/A':>15} {'N/A':>12}")
lines.append("")

for model in MODELS:
    lines.append(f"## {model.upper()} results (full-ranking against all catalog items)")
    header = f"{'dataset':<15}" + "".join(f"{m:>12}" for m in METRICS)
    lines.append(header)
    lines.append("-" * (15 + 12 * len(METRICS)))
    for ds in DATASETS:
        r = load_results(model, ds)
        if r:
            row = f"{ds:<15}" + "".join(f"{r.get(m, float('nan')):>12.4f}" for m in METRICS)
        else:
            row = f"{ds:<15}" + "".join(f"{'N/A':>12}" for m in METRICS)
        lines.append(row)
    lines.append("")

lines.append("## notes")
log_path = BASE / 'run.log'
if log_path.exists():
    with open(log_path) as f:
        errors = [l.strip() for l in f if 'ERROR' in l or 'FAILED' in l]
    if errors:
        for e in errors[:30]:
            lines.append(f"  {e}")
    else:
        lines.append("  no errors found")
else:
    lines.append("  run.log not found")

lines.append("")
lines.append("=" * 80)

with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"summary written to {OUT_PATH}")
print('\n'.join(lines))
