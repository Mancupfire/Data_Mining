#!/usr/bin/env bash
# Full pipeline: download -> preprocess -> train -> summarize
# Datasets: beauty, scientific, sports (Amazon Review)
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
LOG="$BASE/run.log"
cd "$BASE"
source venv/bin/activate

echo "============================================================" | tee -a "$LOG"
echo "start: $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

python download_data.py 2>&1 | tee -a "$LOG"
python preprocess.py --datasets beauty scientific sports 2>&1 | tee -a "$LOG"

for DS in beauty scientific sports; do
    echo "--- sasrec $DS ---" | tee -a "$LOG"
    python train_sasrec.py --dataset "$DS" --epochs 200 --batch_size 256 2>&1 | tee -a "$LOG"
done

for DS in beauty scientific sports; do
    echo "--- gru4rec $DS ---" | tee -a "$LOG"
    python train_gru4rec.py --dataset "$DS" --epochs 200 --batch_size 256 2>&1 | tee -a "$LOG"
done

python summarize.py 2>&1 | tee -a "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "done: $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
