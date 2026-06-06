# ETEGRec — Run & Reproducibility Notes

Working notes for finalizing the ETEGRec experiments in this repo.

> **Dataset naming.** This repo ships Amazon-2023 categories, **not** Beauty/Sports.
> Per project decision the requested experiments are mapped as:
> - **"Beauty"  → `scientific`** (primary; has a config, a completed run, and `item_features.npy`)
> - **"Sports"  → `game`** (second dataset; config added in this work)
> `instrument` is also present if a third dataset is wanted.

---

## 0. Architecture in one paragraph

ETEGRec = **RQ-VAE item tokenizer** (`vq.RQVAE`) + **T5 generative recommender**
(`model.Model` wrapping `T5ForConditionalGeneration`), trained end-to-end with
**alternating optimization** (`trainer.Trainer`): even epochs update the tokenizer
(`_train_epoch_id`), odd epochs update the recommender (`_train_epoch_rec`), governed
by `cycle`. `main.py` runs **train → finetune → test** in a single invocation.
There is **no** TIER baseline and **no** Q-VAE re-ranking module in this codebase.

---

## 1. Environment assumptions

- One GPU. All commands below pin **`CUDA_VISIBLE_DEVICES=7`**.
- `accelerate_config_ddp.yaml` is already set to single-process (`distributed_type: 'NO'`, `num_processes: 1`).
- Python deps required (installed into the active env during this work):
  `torch`, `accelerate`, `transformers`, `faiss-cpu`, `colorama`, `tqdm`, `scikit-learn`, `pyyaml`, `numpy`.
  Install any missing with: `pip install accelerate transformers faiss-cpu colorama`.
- `python -c "import main"` must succeed before running anything.

### ⚠️ Known blocker — corrupted embedding files
All semantic-embedding `.npy` files are currently **truncated** (≈4 MB each vs the
expected ≈26 MB) and cannot be loaded:
```
dataset/scientific/scientific_emb_256.npy   (expects (25848,256))
dataset/game/game_emb_256.npy               (expects (25612,256))
dataset/instrument/instrument_emb_256.npy   (expects (24587,256))
dataset/scientific/item_features.npy        (expects (25849,256))
```
The RQ-VAE `.pth` checkpoints are intact. **Re-download/copy the full `*_emb_256.npy`
(and `item_features.npy`) from the project's Google Drive before training/eval.**
Sanity check after re-providing:
```bash
python - <<'PY'
import numpy as np, json
for d in ["scientific","game"]:
    a = np.load(f"dataset/{d}/{d}_emb_256.npy")
    n = len(json.load(open(f"dataset/{d}/{d}.emb_map.json")))
    print(d, a.shape, "items_in_map", n, "OK" if a.shape[0]==n else "MISMATCH")
PY
```

---

## 2. Beauty (== `scientific`) — full pipeline

`run.sh` already targets `scientific` (with the optional feature-injection flags).
Recommended single-GPU invocation with logging:
```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/scientific.yaml \
  --lr_rec=0.005 --lr_id=0.0001 --cycle=2 --eval_step=2 \
  --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 \
  --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003 \
  --use_features=True --num_features=256 --item_feature_path=item_features.npy \
  2>&1 | tee logs/etegrec_scientific_$(date +"%Y%m%d_%H%M%S").log
```
- To run **without** the feature adapter, drop the last three flags.
- `main.py` runs train → finetune → test; final `Test Results: {... ndcg@10 ...}` is printed and logged.
- Checkpoints: `./myckpt/scientific/<timestamp>/<epoch>.pt` (+ `.pt.rqvae` + `.code.json`).

## 3. Sports (== `game`) — full pipeline

Added in this work: `config/game.yaml` (mirrors `scientific.yaml`) and `run_game.sh`.
`game` has no `item_features.npy`, so it runs without the feature adapter.
```bash
bash run_game.sh
# equivalently:
mkdir -p logs
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/game.yaml \
  --lr_rec=0.005 --lr_id=0.0001 --cycle=2 --eval_step=2 \
  --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 \
  --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003 \
  2>&1 | tee logs/etegrec_game_$(date +"%Y%m%d_%H%M%S").log
```

### Smoke test (after data is fixed)
Validate the full path cheaply before a long run by overriding a few flags:
```bash
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/game.yaml --epochs=2 --eval_step=2 --warmup_steps=10 --early_stop=2 \
  2>&1 | tee logs/smoke_game_$(date +"%Y%m%d_%H%M%S").log
```

---

## 4. K-means codebook (status + how to enable)

**Already implemented — not a missing feature.** The RQ-VAE codebook can be
initialized with k-means:
- `layers.py:kmeans()` (scikit-learn `KMeans`) is called by `vq.VectorQuantizer.init_emb`
  when `kmeans_init: True`; it runs on the first training batch.
- `utils.py:kmeans()` (faiss) is used by the standalone RQ-VAE tokenization pipeline.
- Toggle via config: `kmeans_init` / `kmeans_iters` (currently `kmeans_init: False`).

Where it actually matters: **RQ-VAE pretraining** (`RQVAE/run_pretrain.sh` already passes
`--kmeans_init True`). In the end-to-end `main.py` path, `model_id` is loaded from the
pretrained `rqvae_path`, so the init scheme is inherited from pretraining.

To regenerate the codebook from scratch with k-means init:
```bash
cd RQVAE
CUDA_VISIBLE_DEVICES=7 bash run_pretrain.sh        # uses --kmeans_init True
cd ..
# then point config['rqvae_path'] at the new checkpoint
```

## 5. Codebook utilization analysis (before / after)

New script: `scripts/analyze_codebook_utilization.py`. Reports, per RQ-VAE level:
`total_codes`, `used_codes`, `utilization_rate`, `dead_codes`, `assignment_entropy`
(+ normalized), `top_10_most_used_codes`, `bottom_10_least_used_codes`, plus joint
semantic-id `collision_rate`. Writes to `outputs/codebook_utilization_{dataset}.json`
(merges `before`/`after` tags into one file).

```bash
# offline validation of the metric code path (no data needed)
python scripts/analyze_codebook_utilization.py --self_test

# BEFORE training: pretrained RQ-VAE (needs valid *_emb_256.npy)
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_codebook_utilization.py \
  --config ./config/scientific.yaml --tag before

# AFTER training: tokenizer saved during the end-to-end run
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_codebook_utilization.py \
  --config ./config/scientific.yaml --tag after \
  --rqvae_ckpt ./myckpt/scientific/<run>/<best_epoch>.pt.rqvae
```
(Same with `--config ./config/game.yaml` for Sports.)

---

## 6. TIER baseline — NOT in this repo

No TIER baseline code, and no Sentence-BERT embeddings exist (semantic embeddings here
are **SASRec**, not SBERT). Per project decision this was scoped out (verify/document
only). If wanted later, it would be a new component, not a wiring change.

## 7. Q-VAE re-ranking ablation — NOT in this repo

There is no Q-VAE re-ranking module and no candidate-reranking path (inference is beam
search over RQ codes in `model.Model.generate`). The requested
`outputs/ablation_qvae_rerank_{dataset}.csv` cannot be produced without first building
such a module; scoped out per project decision. The +2–3% NDCG@10 target therefore
**remains unverified** — see blockers below.

---

## 8. Where logs / results / checkpoints live

| Artifact | Location |
|---|---|
| Training logs (run scripts) | `./logs/etegrec_<dataset>_<timestamp>.log` |
| Framework logger output | `./logs/<dataset>/<run_id>.log` |
| Recommender + tokenizer checkpoints | `./myckpt/<dataset>/<timestamp>/<epoch>.pt(.rqvae)` |
| Per-epoch semantic-id codes | `./myckpt/<dataset>/<timestamp>/<epoch>.code.json` |
| Pretrained RQ-VAE | `./dataset/<dataset>/256-512-256-128.rqvae.pth` |
| RQ-VAE pretrain checkpoints | `./RQVAE/rqvae_ckpt/<dataset>/...` |
| Codebook utilization report | `./outputs/codebook_utilization_<dataset>.json` |

---

## 9. Known issues / TODOs

1. **[BLOCKER] Corrupted embeddings** — all `*_emb_256.npy` + `item_features.npy` are
   truncated (~4 MB). Must be re-provided before any training / eval / real codebook
   analysis. Everything else is ready and import-clean.
2. **Beauty/Sports are mapped** to `scientific`/`game`; there is no literal
   Amazon Beauty/Sports data in the repo.
3. **TIER** baseline and **Q-VAE re-ranking** ablation do not exist (scoped out). The
   +2–3% NDCG@10 re-ranking claim is therefore not verifiable in the current codebase.
4. `transformers` installed here is a recent major version (5.x); the `main.py` import
   chain succeeds, but if a long run hits an API incompatibility, pin an older
   transformers (the paper used `transformers` compatible with `torch==2.4.0+cu121`).
5. After data is restored, run the smoke test (§3) before committing to a full run.
