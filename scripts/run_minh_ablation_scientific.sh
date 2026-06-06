#!/usr/bin/env bash
# ============================================================================
# Minh's ablation study — SCIENTIFIC dataset only.
# Removes one SEA-Rec loss/component at a time and trains end-to-end.
#
# IMPORTANT (honesty / labeling):
#   * The HEADLINE "Full SEA-Rec" number is the already-COMPLETED full-schedule
#     run (log etegrec_scientific_20260604_192028.log, Test ndcg@10=0.02186).
#   * The runs below use a REDUCED, FIXED budget shared by ALL variants so they
#     are mutually comparable. They are labelled run_type = "short" everywhere.
#     DO NOT compare a short row directly against the full-schedule headline.
#   * The KL / decoder-contrastive alignment losses act in the cyclic
#     co-training (pretrain) phase; finetune uses code_loss only. So the budget
#     keeps a real pretrain phase (warm_epoch small) for the ablation to bite.
#
# Runs SEQUENTIALLY on one GPU (CUDA device 7). Never in parallel.
# Each variant writes its own tee'd log + a .done marker on success.
# ============================================================================
set -u
cd "$(dirname "$0")/.."          # repo root
mkdir -p logs outputs

GPU=${GPU:-7}
CFG=./config/scientific.yaml

# ---- shared reduced budget (edit here to lengthen/shorten the whole sweep) ----
PRE_EPOCHS=${PRE_EPOCHS:-15}        # pretrain (cyclic co-training) cap
PRE_EARLY=${PRE_EARLY:-5}
WARM=${WARM:-2}                      # when KL/code/dec_cl losses switch on in pretrain
FT_EPOCHS=${FT_EPOCHS:-15}           # finetune cap  (new backward-compat config key)
FT_EARLY=${FT_EARLY:-5}
EVAL_STEP=${EVAL_STEP:-2}

COMMON="--config ${CFG} \
  --lr_rec=0.005 --lr_id=0.0001 --cycle=2 \
  --epochs=${PRE_EPOCHS} --early_stop=${PRE_EARLY} --warm_epoch=${WARM} --eval_step=${EVAL_STEP} \
  --finetune_epochs=${FT_EPOCHS} --finetune_early_stop=${FT_EARLY} --finetune_eval_step=${EVAL_STEP}"

run_variant () {
  local name="$1"; shift
  local extra="$*"
  local ts; ts=$(date +"%Y%m%d_%H%M%S")
  local logf="logs/ablation_scientific_${name}_${ts}.log"
  echo "[$(date '+%F %T')] >>> START ablation '${name}'  ->  ${logf}"
  echo "    extra flags: ${extra}"
  CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --config_file accelerate_config_ddp.yaml main.py \
      ${COMMON} ${extra} 2>&1 | tee "${logf}"
  local rc=${PIPESTATUS[0]}
  if [ "${rc}" -eq 0 ]; then
    touch "${logf}.done"
    echo "[$(date '+%F %T')] <<< DONE  ablation '${name}' (rc=0)"
  else
    echo "[$(date '+%F %T')] <<< FAILED ablation '${name}' (rc=${rc}) — continuing to next"
  fi
}

echo "############ Minh scientific ablation sweep (run_type=short) ############"
echo "budget: pretrain ${PRE_EPOCHS}/early ${PRE_EARLY}/warm ${WARM} ; finetune ${FT_EPOCHS}/early ${FT_EARLY} ; eval_step ${EVAL_STEP}"

# A. Full SEA-Rec (short baseline: all losses + feature adapter)
run_variant "full_short" \
  --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003 \
  --use_features=True --num_features=256 --item_feature_path=item_features.npy

# B. w/o KL alignment loss (keep dec-CL + features)
run_variant "no_kl" \
  --rec_kl_loss=0 --rec_dec_cl_loss=0.0003 --id_kl_loss=0 --id_dec_cl_loss=0.0003 \
  --use_features=True --num_features=256 --item_feature_path=item_features.npy

# C. w/o decoder contrastive loss (keep KL + features)
run_variant "no_deccl" \
  --rec_kl_loss=0.0001 --rec_dec_cl_loss=0 --id_kl_loss=0.0001 --id_dec_cl_loss=0 \
  --use_features=True --num_features=256 --item_feature_path=item_features.npy

# D. w/o ALL alignment losses (KL + dec-CL off; keep features)
run_variant "no_align" \
  --rec_kl_loss=0 --rec_dec_cl_loss=0 --id_kl_loss=0 --id_dec_cl_loss=0 \
  --use_features=True --num_features=256 --item_feature_path=item_features.npy

# E. w/o feature adapter (keep KL + dec-CL)
run_variant "no_features" \
  --rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003 --id_kl_loss=0.0001 --id_dec_cl_loss=0.0003 \
  --use_features=False

echo "############ ablation sweep finished ############"
