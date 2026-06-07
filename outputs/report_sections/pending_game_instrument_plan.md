# Pending Additional Dataset Experiments — Game & Instrument (Minh)

_These are **pending experiments, not optional ones.** Scientific is the only dataset with a
completed full run. Game and Instrument data are present, but neither has a completed result, and
Instrument is additionally missing its config. This section is the plan to finish them._

## Current state (verified on disk, 2026-06-07)

| Dataset | Data present | Config present | Completed run / logs | Feature adapter | Status |
|---|---|---|---|---|---|
| Scientific | ✓ | ✓ `config/scientific.yaml` | ✓ `logs/etegrec_scientific_20260604_192028.log` | ✓ has `item_features.npy` | **DONE** |
| **Game** | ✓ (`game_emb_256.npy`, `*.jsonl`, `256-512-256-128.rqvae.pth`) | ✓ `config/game.yaml` | ✗ **no Game log in `logs/`** | ✗ no `item_features.npy` | **PENDING** |
| **Instrument** | ✓ (`instrument_emb_256.npy`, `*.jsonl`, `256-512-256-128.rqvae.pth`) | ✗ **`config/instrument.yaml` missing** | ✗ none | ✗ no `item_features.npy` | **PENDING** |

Notes:
- `config/` contains only `game.yaml` and `scientific.yaml` — **no `instrument.yaml`**.
- `logs/` contains only Scientific runs/ablations — **no Game or Instrument run**.
- Only Scientific has `item_features.npy`, so only Scientific uses the feature adapter
  (`--use_features`). Game and Instrument run **without** the adapter.

## Game — plan
Everything needed exists; just run and parse.
```bash
# Full Game run (no feature adapter — game has no item_features.npy):
bash run_game.sh                       # pinned to GPU 7, writes logs/etegrec_game_<ts>.log

# After it finishes, refresh parsed results:
python scripts/parse_training_results.py     # fills the Game row in outputs/minh_*results*

# Codebook after-training visualization for Game:
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_codebook_utilization.py \
  --config ./config/game.yaml --tag after \
  --rqvae_ckpt <best_epoch>.pt.rqvae \
  --output outputs/codebook_utilization_game_after.json
CUDA_VISIBLE_DEVICES="" python scripts/plot_codebook_utilization.py --dataset game --device cpu
```
This also upgrades the Game codebook figures from **before-only placeholders** to real
before-vs-after (see `codebook_figure_captions.md`).

## Instrument — plan
First create the missing config, then run like Game.
```bash
# 1) Create config/instrument.yaml by cloning game.yaml and swapping dataset/paths:
#      dataset: instrument
#      semantic_emb_path: instrument_emb_256.npy
#      rqvae_path: ./dataset/instrument/256-512-256-128.rqvae.pth
#    (keep map_path: .emb_map.json; all other hyper-params identical to game.yaml)

# 2) Run (no feature adapter — instrument has no item_features.npy):
TS=$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/instrument.yaml \
  --lr_rec=0.005 --lr_id=0.0001 --cycle=2 --eval_step=2 \
  --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 \
  --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003 \
  2>&1 | tee logs/etegrec_instrument_${TS}.log

# 3) Parse + codebook viz (same as Game, with --config ./config/instrument.yaml).
python scripts/parse_training_results.py
```

## Definition of done
- [ ] Game: completed `logs/etegrec_game_*.log` + final test row parsed into
      `outputs/minh_results_table.md` / `outputs/minh_ablation_results.*`.
- [ ] Game: `outputs/codebook_utilization_game_after.json` + real before-vs-after figures.
- [ ] Instrument: `config/instrument.yaml` created.
- [ ] Instrument: completed `logs/etegrec_instrument_*.log` + parsed test row.
- [ ] Instrument: codebook before (and after) utilization + figures.

## Constraints
- Single GPU (CUDA 7); full-schedule runs are ~8 h each — run sequentially.
- Do **not** add `--use_features` for Game/Instrument (no feature file exists for them).
