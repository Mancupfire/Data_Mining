# SMOKE TEST & FULL-TRAINING COMMANDS

Date: 2026-06-04. GPU: CUDA device 7. Logs in `./logs/`.

## Smoke test result (scientific, 1 epoch)
**PASS — the training + full-ranking evaluation pipeline runs end to end.** Verified:
- Data load (259,992 train rows), item-feature load `(25849,256)`, RQ-VAE load (no missing keys).
- Pretrain epoch 0 forward+backward OK: `loss 0.0085, vq_loss 0.0048, code_loss 5.56, kl_loss 0.055, dec_cl_loss 12.24` (~77–103 s/epoch).
- Per-level code-balance eval + **generative full-ranking eval** produced `Val Results: recall@{1,5,10}, ndcg@{5,10}` (all 0.0 — expected after a single epoch).

### Bug fixed during smoke (real, affects every run)
`metrics.py` used `np.asfarray`, **removed in NumPy 2.0** → evaluation crashed. Fixed to `np.asarray(..., dtype=float)` (2 lines). Without this, no run can evaluate.

### Known non-issue
After the degenerate 1-epoch run, `trainer.test()` crashes with `torch.load(None)` because no "best" checkpoint was saved — validation never beat the 0.0 init (`trainer.py:540` only saves on improvement). In a real multi-epoch run validation exceeds 0, a best checkpoint is saved, and this path works. **Not a blocker for full training.**

## Full-training command — scientific (Beauty-analog)
```bash
cd /mnt/disk1/backup_user/minh.ntn/ETEGRec
mkdir -p logs
TS=$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
    --config ./config/scientific.yaml \
    --lr_rec=0.005 --lr_id=0.0001 --cycle=2 --eval_step=2 \
    --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003 \
    --use_features=True --num_features=256 \
    --item_feature_path=item_features.npy \
    2>&1 | tee logs/etegrec_scientific_${TS}.log
```
(`item_feature_path` is resolved relative to `dataset/scientific/` by `main.py:52`; the file was regenerated via `python generate_features.py`.)

## Full-training command — game (Sports-*placeholder*, actually Video_Games)
```bash
cd /mnt/disk1/backup_user/minh.ntn/ETEGRec
mkdir -p logs
TS=$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
    --config ./config/game.yaml \
    --lr_rec=0.005 --lr_id=0.0001 --cycle=2 --eval_step=2 \
    --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003 \
    2>&1 | tee logs/etegrec_game_${TS}.log
```
(`game` does NOT use `--use_features` — there is no `item_features.npy` for game; `game.yaml` has no feature settings. Equivalent to `bash run_game.sh`.)

## Cost / time note
Pretrain epoch ≈ 77–103 s (scientific, bs 512). Per-epoch validation = generative beam search (`num_beams=20`) over ~51K (scientific) / ~95K (game) users — this dominates wall-clock. Default schedule is up to 400 pretrain + 100 finetune epochs with early stopping (`early_stop` 15 / 10). Game is ~2× larger (95K users, 530K train rows) → expect roughly 2× the scientific wall-clock. Run under `tee` (or `nohup`/tmux) since full runs are long.
