# Item Embedding Provenance (Minh)

_Purpose: state exactly what the 256-d item embeddings are and where they came from, to correct
any "Sentence-BERT / text-content embedding" assumption._

## TL;DR
The 256-dimensional per-item vectors used by ETEGRec/SEA-Rec are **SASRec collaborative item
embeddings**, taken **already-computed** from the **ETEGRec authors' preprocessed Google Drive
release**. They are **not** Sentence-BERT (SBERT) embeddings and **not** text/content embeddings
generated locally.

## Evidence in this repo
- **`README.md`** (authors' instructions): "You can download the **SASRec embeddings**, pretrained
  RQVAE weights and interaction data used in our paper from
  [Google Drive](https://drive.google.com/drive/folders/1KiPpB7uq7eFc4qB74cFOxhtY3H8nWgAI)."
- **`DATASET_PROVENANCE_AND_COUNTS.md`**: each dataset is "already **fully preprocessed**
  (leave-one-out splits + **SASRec 256-d item embeddings** + pretrained RQ-VAE)"; the data was
  "obtained **pre-processed from the paper authors**, not produced by a local Amazon-2023 pipeline."
- **`RUN_ETEGREC_NOTES.md`**: "No Sentence-BERT embeddings exist (semantic embeddings here are
  **SASRec**, not SBERT)."
- **Files on disk** (`numpy` shapes/dtypes verified):
  | file | shape | dtype |
  |---|---|---|
  | `dataset/scientific/scientific_emb_256.npy` | (25848, 256) | float32 |
  | `dataset/game/game_emb_256.npy` | (25612, 256) | float32 |
  | `dataset/instrument/instrument_emb_256.npy` | (24587, 256) | float32 |
  | `dataset/scientific/item_features.npy` | (25849, 256) | float32 |
- **Configs** (`config/scientific.yaml`, `config/game.yaml`): `semantic_emb_path: <ds>_emb_256.npy`,
  `semantic_hidden_size: 256` — the RQ-VAE consumes these 256-d vectors directly.

## What this means for the report
1. The RQ-VAE codebooks (k-means-style snap) quantize **SASRec collaborative** vectors, so the
   "semantic IDs" encode **collaborative-filtering structure**, not text semantics.
2. The Scientific **feature adapter** input, `item_features.npy`, is the **same SASRec embedding**
   matrix with a zero PAD row prepended (`generate_features.py` → (25849, 256)). So the "feature"
   pathway injects strong collaborative signal — relevant to the inconclusive Feature-Adapter
   ablation in `ablation_results.md`: the collaborative embeddings are already strong, leaving the
   extra adapter little headroom in a short run.
3. Anywhere the report previously implied SBERT/content embeddings, replace with **SASRec
   collaborative embeddings (authors' Google Drive preprocessed release)**.

## How to re-verify
```bash
python - <<'PY'
import numpy as np
for d in ["scientific","game","instrument"]:
    a = np.load(f"dataset/{d}/{d}_emb_256.npy", mmap_mode="r")
    print(d, a.shape, a.dtype)
PY
grep -n "SASRec" README.md DATASET_PROVENANCE_AND_COUNTS.md RUN_ETEGREC_NOTES.md
```
