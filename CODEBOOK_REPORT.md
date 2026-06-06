# CODEBOOK UTILIZATION REPORT

Date: 2026-06-04. Tokenizer = residual VQ-VAE (`vq.RQVAE`), 3 levels × 256 codes.

## K-means init — status
- **Implemented:** yes. `vq.VectorQuantizer.init_emb()` calls `utils.kmeans` (faiss); gated by `kmeans_init`/`kmeans_iters`. Runs on the first training batch when `kmeans_init: True`.
- **Enabled:** **no** — both `config/scientific.yaml` and `config/game.yaml` set `kmeans_init: False`. The shipped pretrained RQ-VAE was trained with random init.
- **To enable:** add `--kmeans_init=True` (and optionally `--kmeans_iters=100`) to the RQ-VAE pretraining command, or set it in the config, then re-pretrain and point `rqvae_path` at the new checkpoint.

## BEFORE measurement (pretrained RQ-VAE shipped with the data)
Script: `scripts/analyze_codebook_utilization.py` (ran on GPU 7). Outputs:
`outputs/codebook_utilization_scientific_before.json`, `outputs/codebook_utilization_game_before.json`.

| dataset | level | used/total | utilization | dead | norm. entropy |
|---|---|---|---|---|---|
| scientific | 0 | 256/256 | 100.0% | 0 | 0.971 |
| scientific | 1 | 256/256 | 100.0% | 0 | 0.997 |
| scientific | 2 | 256/256 | 100.0% | 0 | 0.998 |
| game | 0 | 254/256 | 99.2% | 2 | 0.979 |
| game | 1 | 256/256 | 100.0% | 0 | 0.994 |
| game | 2 | 256/256 | 100.0% | 0 | 0.995 |

Joint (3-tuple) semantic-ID collisions:
- scientific: 25,454 unique / 25,848 items → **collision rate 1.52%**
- game: 25,166 unique / 25,612 items → **collision rate 1.74%**

**Read:** the pretrained codebooks are already near-ideal — ~100% utilization, almost no dead codes, high normalized entropy (≈0.97–1.0, i.e. close to uniform code usage), and low collision. There is little headroom for k-means init to improve utilization *before* training; its value would be mainly faster/more stable convergence, not higher final utilization.

## AFTER measurement (end-to-end trained tokenizer) — DONE (2026-06-05)
The full Scientific run completed (best epoch 62, `myckpt/scientific/Jun-04-2026_19-21-4ee3de/62.pt.rqvae`). Measured with:
```bash
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_codebook_utilization.py \
  --config ./config/scientific.yaml --tag after \
  --rqvae_ckpt ./myckpt/scientific/Jun-04-2026_19-21-4ee3de/62.pt.rqvae \
  --output outputs/codebook_utilization_scientific_after.json
```

| dataset | level | used/total | utilization | dead | norm. entropy (before→after) |
|---|---|---|---|---|---|
| scientific | 0 | 256/256 | 100.0% | 0 | 0.971 → **0.953** |
| scientific | 1 | 256/256 | 100.0% | 0 | 0.997 → **0.970** |
| scientific | 2 | 256/256 | 100.0% | 0 | 0.998 → **0.987** |

Joint collision rate: **1.52% → 1.77%** (25,454 → 25,390 unique semantic IDs / 25,848 items).

**Read:** end-to-end training does **not** collapse the codebook (still 100% utilization, 0 dead
codes), but code usage becomes **less uniform** (entropy drops at every level) and collisions
rise slightly — the tokenizer specializes its codes for the recommendation objective. Plots:
`outputs/figures/codebook_{utilization,dead_codes,entropy,usage_hist_level0,top_codes_level0}_scientific_before_after.png`
(`scripts/plot_codebook_utilization.py`). Game AFTER is pending the full game run.
