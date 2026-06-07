# Codebook Figure Captions (Minh)

_Ready-to-paste captions for the codebook / k-means visualization figures in `outputs/figures/`.
All figures plot the **shipped random-init RQ-VAE** (`kmeans_init: False`) quantizing **SASRec
256-d collaborative embeddings** (see `embedding_provenance.md`). Numbers come from
`outputs/codebook_utilization_scientific_{before,after}.json` and `..._game_before.json`._

## Scope note (read first)
- **Scientific** figures show genuine **before vs. after** end-to-end training (both checkpoints
  exist).
- **Game** figures are **before-only**: despite the `_before_after` filename, the "after" series is
  a placeholder because **no completed Game run exists yet** (see
  `pending_game_instrument_plan.md`). Caption them as before-only until the Game run finishes.
- These figures are the **before-vs-after visualization**, which is **complete**. They are **not**
  the random-vs-k-means initialization ablation, which is **future work**
  (`kmeans_ablation_future_work.md`).

## Scientific figures
- **`codebook_utilization_scientific_before_after.png`** — *Codebook utilization per RQ-VAE level
  (Scientific), before vs. after end-to-end training.* All three levels use **256/256 codes
  (100%)** both before and after training — no dead codes appear during joint training.
- **`codebook_dead_codes_scientific_before_after.png`** — *Dead (never-assigned) codes per level
  (Scientific).* **Zero dead codes** at every level, before and after — the tokenizer does not
  collapse under joint training.
- **`codebook_entropy_scientific_before_after.png`** — *Normalized usage entropy per level
  (Scientific).* Entropy **decreases at every level** after training (L0 0.971→0.953, L1
  0.997→0.970, L2 0.998→0.987): codes **specialize** away from uniform usage toward the long-tail
  item distribution.
- **`codebook_usage_hist_level0_scientific_before_after.png`** — *Level-0 code-usage histogram
  (Scientific).* **Before:** tightly peaked (~110–160 items/code, near-uniform). **After:** spreads
  out — more lightly-used codes plus a heavy-usage tail (up to ~340 items/code).
- **`codebook_top_codes_level0_scientific_before_after.png`** — *Most-used level-0 codes
  (Scientific), before vs. after.* The top codes carry more mass after training, visualizing the
  emergence of "popular-region" codes.

> Scientific summary stat for captions: **joint semantic-ID collision rate 1.52% → 1.77%**
> (25,454 → 25,390 unique IDs out of 25,848 items).

## Game figures (before-only placeholder)
- **`codebook_utilization_game_before_after.png`** — *Codebook utilization per level (Game),
  pretrained RQ-VAE (before only).* Level 0 = 254/256 (99.2%, 2 dead); levels 1–2 = 256/256 (100%).
  *After-training series pending the Game run.*
- **`codebook_dead_codes_game_before_after.png`** — *Dead codes per level (Game), before only.*
  2 dead codes at level 0; 0 at levels 1–2.
- **`codebook_entropy_game_before_after.png`** — *Normalized usage entropy per level (Game),
  before only.* 0.979 / 0.994 / 0.995.
- **`codebook_usage_hist_level0_game_before_after.png`** — *Level-0 usage histogram (Game),
  before only.*
- **`codebook_top_codes_level0_game_before_after.png`** — *Most-used level-0 codes (Game),
  before only.*

> Game summary stat: collision rate 1.74% (before). Replace "before only" captions once the Game
> full run produces an after-training checkpoint.
