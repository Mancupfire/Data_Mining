#!/usr/bin/env bash
# Pipeline for ETEGRec datasets (game, instrument, scientific).
# Run convert_etegrec.py first to generate data/{game,instrument,scientific}/.
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"
PYTHON="$BASE/venv/bin/python"

# Convert ETEGRec format if needed
if [ ! -f data/game/train.txt ]; then
    echo "converting ETEGRec datasets..."
    $PYTHON convert_etegrec.py
fi

for DS in game instrument scientific; do
    echo "--- sasrec $DS ---" | tee -a "logs/sasrec_${DS}.log"
    $PYTHON train_sasrec.py --dataset $DS --epochs 200 --batch_size 256 --lr 1e-3 --max_len 50 \
        2>&1 | tee -a "logs/sasrec_${DS}.log"
done

for DS in game instrument scientific; do
    echo "--- gru4rec $DS ---" | tee -a "logs/gru4rec_${DS}.log"
    $PYTHON train_gru4rec.py --dataset $DS --epochs 200 --batch_size 256 --lr 1e-3 --max_len 50 \
        2>&1 | tee -a "logs/gru4rec_${DS}.log"
done

echo "all done, generating summary..."
$PYTHON summarize.py
