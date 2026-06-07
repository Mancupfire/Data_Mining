# K-means / Codebook Visualization — Before vs After Training (Minh)

_Figures in `outputs/figures/`. Numbers from `outputs/codebook_utilization_scientific_{before,after}.json` (and `..._game_before.json`), produced by `scripts/analyze_codebook_utilization.py`; plotted by `scripts/plot_codebook_utilization.py`._

## What the codebook does (for non-experts)
> **Embedding source:** the 256-d item vectors here are **SASRec collaborative embeddings** from
> the **ETEGRec authors' preprocessed Google Drive release** (`*_emb_256.npy`) — **not
> Sentence-BERT / text-content embeddings**. See `embedding_provenance.md`.

ETEGRec/SEA-Rec does **not** feed raw item IDs to the recommender. Each item's 256-d
SASRec collaborative embedding is passed through a **Residual-Quantized VAE (RQ-VAE)**, which turns it
into a short sequence of **discrete codes** — here **3 levels × 256 codes** (a 4th code
disambiguates collisions). Each level has its own **codebook**: 256 learnable vectors
("centroids"). An item is encoded by, at each level, snapping the residual to the nearest
codebook vector — exactly like a **k-means assignment**. So an item becomes a *semantic ID*
such as `(35, 12, 240)`. The T5 recommender then learns to **generate** these code
sequences instead of scoring tens of thousands of raw IDs.

**Why k-means matters here:** the codebooks can be **initialized** with k-means on the item
embeddings (`kmeans_init`). Random initialization can leave many codes unused
("dead codes") or cause **code collapse** (most items snap to a few codes); k-means
initialization starts the centroids on real embedding clusters, which improves stability
and convergence. In this repo the shipped RQ-VAE was pretrained with **random init**
(`kmeans_init: False`), and we measured utilization before vs after end-to-end training.

> **Scope of this section:** the **before-vs-after utilization visualization is COMPLETE** (below).
> A **dedicated random-init vs. k-means-init ablation is NOT done** — that requires re-pretraining
> the RQ-VAE with `kmeans_init: True`. It is **future work**, scoped in
> `kmeans_ablation_future_work.md`. The two are different deliverables: this section *visualizes*
> the shipped random-init codebook; the ablation would *compare* two init strategies head-to-head.

## Utilization metrics we report
- **Used / dead codes** — how many of the 256 codes per level are ever assigned (dead = never used).
- **Normalized entropy** — how *uniform* the usage is (1.0 = every code used equally; lower = a few codes dominate).
- **Collision rate** — fraction of items that, after the 3 levels, still share an identical semantic ID (need the 4th tie-break code).

## Results — Scientific (before vs after end-to-end training)

| level | used/total (before→after) | dead (before→after) | norm. entropy (before→after) |
|---|---|---|---|
| 0 | 256/256 → 256/256 | 0 → 0 | **0.971 → 0.953** |
| 1 | 256/256 → 256/256 | 0 → 0 | **0.997 → 0.970** |
| 2 | 256/256 → 256/256 | 0 → 0 | **0.998 → 0.987** |

Joint semantic-ID **collision rate: 1.52% → 1.77%** (25,454 → 25,390 unique IDs out of 25,848 items).

**Game (before only — full game run pending):** level 0 = 254/256 used (99.2%, 2 dead),
levels 1–2 = 256/256 (100%); normalized entropy 0.979 / 0.994 / 0.995; collision 1.74%.
_After-training game visualization is pending the full game run._

## Figures
- `codebook_utilization_scientific_before_after.png` — used codes per level (100% before and after).
- `codebook_dead_codes_scientific_before_after.png` — dead codes per level (0 before and after).
- `codebook_entropy_scientific_before_after.png` — normalized entropy per level (drops at every level after training).
- `codebook_usage_hist_level0_scientific_before_after.png` — level-0 usage histogram: **before** is tightly peaked (~110–160 items/code, near-uniform); **after** spreads out, with more lightly-used codes *and* a heavy-usage tail (up to ~340 items/code).
- `codebook_top_codes_level0_scientific_before_after.png` — most-used level-0 codes before vs after.
- Same set with `_game_` for game (before-only placeholder until its full run finishes).

## Interpretation
1. **No collapse.** End-to-end training keeps **100% utilization with zero dead codes** at all
   three levels — the tokenizer does not degenerate during joint training. This is the main
   safety check.
2. **Codes specialize.** Normalized entropy **decreases** at every level (e.g. 0.971→0.953 at
   level 0) and collisions rise slightly (1.52%→1.77%). Training pushes code usage *away* from
   the pure-uniform RQ-VAE objective toward a distribution that is **more useful for next-item
   generation** — some codes become "popular-region" codes, matching the long-tail item
   distribution shown in the data analysis. The level-0 histogram visualizes this directly.
3. **K-means relevance.** Because utilization was already ~100% with high entropy *before*
   training (random-init RQ-VAE already near-ideal on these data), **k-means init has little
   headroom to raise utilization** here. Its value would be **convergence stability /
   semantic organization**, not higher final utilization. **This is a hypothesis, not a measured
   result** — the dedicated random-vs-k-means ablation is **future work** (see
   `kmeans_ablation_future_work.md`): re-pretrain the RQ-VAE with `kmeans_init: True`
   (`RQVAE/run_pretrain.sh`) and repoint `rqvae_path`.

### How to reproduce
```bash
# BEFORE (pretrained RQ-VAE) — already done
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_codebook_utilization.py --config ./config/scientific.yaml --tag before
# AFTER (best end-to-end checkpoint) — already done
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_codebook_utilization.py --config ./config/scientific.yaml --tag after \
  --rqvae_ckpt ./myckpt/scientific/Jun-04-2026_19-21-4ee3de/62.pt.rqvae \
  --output outputs/codebook_utilization_scientific_after.json
# PLOTS (CPU; safe while GPU is busy)
CUDA_VISIBLE_DEVICES="" python scripts/plot_codebook_utilization.py --dataset scientific --device cpu
```
