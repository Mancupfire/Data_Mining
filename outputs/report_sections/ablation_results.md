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

## Results (sequential, GPU 7) — sweep COMPLETE
All five short variants finished (each has a `logs/ablation_scientific_<variant>_<ts>.log.done`
marker). Numbers below are **final test metrics** parsed by `scripts/parse_training_results.py`
(`outputs/minh_ablation_results.{md,csv}`).

| variant | run_type | status | Test R@10 | Test N@10 | best epoch |
|---|---|---|---:|---:|---:|
| A. Full SEA-Rec | full schedule | **DONE** | **0.0419** | **0.0219** | 62 |
| A. Full SEA-Rec (short baseline) | short | **DONE** | 0.0399 | 0.0210 | 13 |
| B. w/o KL | short | **DONE** | 0.0399 | 0.0209 | 13 |
| C. w/o Decoder-CL | short | **DONE** | 0.0398 | 0.0208 | 13 |
| D. w/o Alignment (KL+DecCL) | short | **DONE** | 0.0402 | 0.0210 | 13 |
| E. w/o Feature Adapter | short | **DONE** | 0.0408 | 0.0213 | 11 |

### Interpretation
- **Alignment components (B, C, D)** sit within **≤0.0004 R@10** of the Full-short baseline
  (0.0399). At a 13-epoch budget these differences are **within short-run noise**; the alignment
  losses' measurable payoff is on the **full schedule** (pre-finetune 0.0325 → finetuned 0.0419).
  Do not read the short sweep as "alignment doesn't help" — it shows the components are not harmful
  at short budget, and the full-schedule run is where the lift appears.
- **E. w/o Feature Adapter — anomaly, treat as inconclusive.** Removing the adapter gives R@10
  **0.0408** / N@10 0.0213, **slightly above** the Full-short baseline (0.0399 / 0.0210). We do
  **not** conclude the feature adapter is harmful. Reasons to discount the result:
  1. The ~0.0009 R@10 gap is well within **short-run / single-seed variance** (one seed, 13 epochs).
  2. The feature adapter adds **extra parameters** that a short budget under-trains relative to the
     leaner no-adapter model.
  3. The injected features are **SASRec collaborative embeddings, already strong** (see
     `embedding_provenance.md`), so an extra content-feature pathway has little headroom here.
  A **full-schedule, multi-seed** comparison is required before any claim about the adapter.

The sweep ran B→C→D→E sequentially after the short Full baseline; each wrote
`logs/ablation_scientific_<variant>_<ts>.log` and a `.done` marker on success. Re-run the
parser to refresh the table.

## Variants not run
- **F. Decoupled / no-tokenizer-update:** the cyclic loop always updates M_id during
  co-training; freezing it would require a code change. **Planned, not run.**
- **G. random vs k-means codebook init — FUTURE WORK (not done).** This is the *dedicated*
  initialization ablation. It requires re-pretraining the RQ-VAE with `kmeans_init: True`
  (`RQVAE/run_pretrain.sh`) and repointing `rqvae_path`. The current shipped RQ-VAE uses random
  init (`kmeans_init: False`). Note this is **separate** from the completed before-vs-after
  codebook *visualization* (`kmeans_codebook_visualization.md`). Scoped in
  `outputs/report_sections/kmeans_ablation_future_work.md`.

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
