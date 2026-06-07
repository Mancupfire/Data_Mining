# recsys

Sequential recommendation baselines: SASRec and GRU4Rec trained and evaluated on Amazon Review and ETEGRec datasets.

Both models are evaluated with full-ranking (scored against the entire item catalog). Metrics: Recall@k and NDCG@k for k = 1, 5, 10.

## Layout

```
recsys/
├── data/
│   ├── etegrec/           # raw ETEGRec JSONL + embeddings (gitignored)
│   │   ├── game/
│   │   ├── instrument/
│   │   └── scientific/
│   ├── beauty/            # processed splits (gitignored)
│   ├── sports/
│   ├── scientific/
│   ├── game/
│   └── instrument/
├── models/
│   ├── sasrec.py          # SASRec (Kang & McAuley, ICDM 2018)
│   └── gru4rec.py         # GRU4Rec (Hidasi et al., ICLR 2016)
├── dataset.py             # Dataset classes and data loading
├── convert_etegrec.py     # Convert ETEGRec JSONL -> baseline format
├── download_data.py       # Download Amazon Review datasets
├── preprocess.py          # 5-core filter, leave-one-out split, re-indexing
├── train_sasrec.py        # Train SASRec
├── train_gru4rec.py       # Train GRU4Rec
├── evaluate.py            # Full-ranking evaluation
├── pipeline.py            # Run all Amazon datasets sequentially
├── summarize.py           # Aggregate results into RESULTS_SUMMARY.txt
├── run_all.sh             # Shell wrapper for the Amazon pipeline
└── run_etegrec_pipeline.sh  # Shell wrapper for the ETEGRec pipeline
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy
```

## Running

**Amazon datasets (beauty, sports, scientific):**

```bash
bash run_all.sh
```

This downloads raw data, preprocesses it, trains both models on all three datasets, and writes `RESULTS_SUMMARY.txt`.

**ETEGRec datasets (game, instrument, scientific):**

Place the ETEGRec JSONL files under `data/etegrec/{game,instrument,scientific}/`, then:

```bash
python convert_etegrec.py   # converts to baseline format
bash run_etegrec_pipeline.sh
```

**Training a single model:**

```bash
python train_sasrec.py --dataset beauty --epochs 200 --batch_size 256
python train_gru4rec.py --dataset game --epochs 200 --batch_size 256
```

**Evaluation only:**

```bash
python evaluate.py --model sasrec --dataset beauty \
    --checkpoint checkpoints/sasrec_beauty_best.pt --split test
```

## Results

Run `python summarize.py` to regenerate `RESULTS_SUMMARY.txt` from whatever result files are in `results/`.

Example results on ETEGRec datasets (full-ranking):

| model   | dataset    | Recall@10 | NDCG@10 |
|---------|------------|-----------|---------|
| SASRec  | game       | 0.022     | 0.011   |
| SASRec  | instrument | 0.024     | 0.012   |
| SASRec  | scientific | 0.014     | 0.007   |
| GRU4Rec | game       | 0.021     | 0.010   |
| GRU4Rec | instrument | 0.024     | 0.012   |
| GRU4Rec | scientific | 0.013     | 0.007   |

## Data sources

- Amazon Review 2023: https://amazon-reviews-2023.github.io
- ETEGRec datasets: provided separately under `data/etegrec/`
