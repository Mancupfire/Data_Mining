# Ablation Study — SEA-Rec components on Scientific (Minh)

_The live, auto-parsed numbers are in `outputs/minh_ablation_results.md` / `.csv` (regenerate any time with `python scripts/parse_training_results.py`). This section explains the design and current status. **No numbers here are fabricated** — pending rows are marked PENDING._

## Design
Each variant removes exactly **one** SEA-Rec component and trains the full pipeline
(cyclic co-training → finetune). The alignment losses (KL, decoder-contrastive) act during the
**cyclic co-training** phase; the feature adapter changes the T5 encoder input.

| variant | KL loss | decoder-CL | feature adapter | what it tests |
|---|:--:|:--:|:--:|---|
| **A. Full SEA-Rec** | ✓ | ✓ | ✓ | full method |
| **B. w/o KL** | ✗ | ✓ | ✓ | value of KL alignment |
| **C. w/o Decoder-CL** | ✓ | ✗ | ✓ | value of decoder contrastive |
| **D. w/o Alignment** | ✗ | ✗ | ✓ | do alignment losses matter at all |
| **E. w/o Feature Adapter** | ✓ | ✓ | ✗ | value of content-feature injection |

## Two budgets (read the labels!)
- **Full schedule (headline):** variant **A** was run to completion on the default schedule
  (pretrain up to 400 / early-stop 15, finetune up to 100 / early-stop 10). This is the
  reported "Full SEA-Rec" headline (Test Recall@10 = **0.0419**, NDCG@10 = **0.0219**).
- **Short controlled sweep (`run_type=short`):** because each full-schedule run takes ~8 h on
  one GPU, the ablation **sweep** (`scripts/run_minh_ablation_scientific.sh`) runs **all five
  variants — including a Full-short baseline — on one identical reduced budget** (pretrain 15 /
  early-stop 5 / warm 2, finetune 15 / early-stop 5, eval every 2). The short rows are
  **only comparable to each other**, NOT to the full-schedule headline. The reduced budget is
  enabled by a backward-compatible `finetune_epochs` / `finetune_early_stop` config key added to
  `trainer.finetune()` (defaults reproduce the original 100/10 behaviour exactly).

## Current status (sequential, GPU 7)
| variant | run_type | status |
|---|---|---|
| A. Full SEA-Rec | full schedule | **DONE** — Test R@10 0.0419 / N@10 0.0219 |
| A. Full SEA-Rec (short baseline) | short | **RUNNING / PENDING** |
| B. w/o KL | short | **PENDING** (queued in sweep) |
| C. w/o Decoder-CL | short | **PENDING** (queued) |
| D. w/o Alignment | short | **PENDING** (queued) |
| E. w/o Feature Adapter | short | **PENDING** (queued) |

The sweep runs B→C→D→E sequentially after the short Full baseline; each writes
`logs/ablation_scientific_<variant>_<ts>.log` and a `.done` marker on success. Re-run the
parser to refresh the table as logs complete.

## Optional variants (documented, not run)
- **F. Decoupled / no-tokenizer-update:** the cyclic loop always updates M_id during
  co-training; freezing it would require a code change. **Planned, not run.**
- **G. random vs k-means codebook init:** requires re-pretraining the RQ-VAE with
  `--kmeans_init True` (`RQVAE/run_pretrain.sh`) and repointing `rqvae_path`. Command prepared;
  **not run** (cost). Current shipped RQ-VAE uses random init.

## How to (re)run
```bash
# full schedule (headline) — already completed once:
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/scientific.yaml --use_features=True --num_features=256 \
  --item_feature_path=item_features.npy  2>&1 | tee logs/etegrec_scientific_$(date +%F_%H%M%S).log

# short controlled ablation sweep (B..E, sequential):
bash scripts/run_minh_ablation_scientific.sh        # GPU=7 by default
python scripts/parse_training_results.py            # refresh outputs/minh_ablation_results.*
```
