#!/usr/bin/env bash
# Sports-analog dataset == "game" (Beauty-analog == "scientific").
# Single-GPU run pinned to CUDA device 7. Logs go under ./logs/.
DATASET=game

mkdir -p logs
TS=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
    --config ./config/${DATASET}.yaml \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --cycle=2 \
    --eval_step=2 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003 \
    2>&1 | tee logs/etegrec_${DATASET}_${TS}.log
