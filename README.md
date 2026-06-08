# SEA-Rec

SEA-Rec is a semantic-enhanced aligned generative sequential recommender built on top of an RQ-VAE item tokenizer and a T5-style encoder-decoder recommender. This repository contains the project code, analysis artifacts, and report for the SEA-Rec course project described in [docs/report/searec_project_report.pdf](/home/hoangnam/Data_Mining/docs/report/searec_project_report.pdf).

## Project Summary

The project studies a simple question: how do we make discrete item codes useful both for reconstruction and for next-item generation?

SEA-Rec keeps the ETEGRec training backbone, then adds:

- cyclic co-training between tokenizer and recommender
- symmetric KL alignment between sequence and target-item code distributions
- decoder-level contrastive alignment
- an optional feature-adapter path that injects 256-d item features into the encoder

The item features used here are SASRec collaborative embeddings from the ETEGRec data release, not text embeddings.

## Main Results

Completed full-schedule SEA-Rec runs reported in the project paper:

| Dataset | Recall@10 | NDCG@10 |
|---|---:|---:|
| Game | 0.0908 | 0.0479 |
| Instrument | 0.0603 | 0.0322 |
| Scientific | 0.0419 | 0.0219 |

Scientific full-catalog comparison against verified baselines:

| Method | Recall@10 | NDCG@10 |
|---|---:|---:|
| BPR-MF | 0.0130 | 0.0064 |
| GRU4Rec | 0.0133 | 0.0069 |
| SASRec | 0.0136 | 0.0072 |
| SEA-Rec | 0.0419 | 0.0219 |

The report also documents that all three RQ-VAE codebook levels remain fully utilized on Scientific after training, with zero dead codes.

## Repository Layout

```text
.
|-- README.md
|-- main.py / model.py / trainer.py / data.py / utils.py / vq.py
|-- config/
|-- RQVAE/
|-- scripts/
|-- baselines/bprmf/
|-- docs/
|   |-- report/searec_project_report.pdf
|   `-- notes/
|-- notebooks/
|   |-- SEARec_Data_Analysis.ipynb
|   `-- exports/
|-- outputs/
`-- dataset/              # ignored; add preprocessed data locally
```

Notes:

- `docs/notes/` stores project notes, provenance checks, validation notes, and codebook analysis writeups.
- `notebooks/exports/` stores rendered notebook artifacts and derived figures.
- `baselines/bprmf/` vendors the BPR-MF baseline code used for Scientific comparisons into this repo.

## Requirements

Core training depends on:

```text
torch
numpy
accelerate
faiss
tqdm
scikit-learn
transformers
pyyaml
```

Notebook/report helper scripts additionally use `pypdf` and `reportlab`.

## Data

This repo does not track the datasets or checkpoints. Put the preprocessed ETEGRec release under `dataset/`, with per-dataset files such as:

- `dataset/scientific/scientific.train.jsonl`
- `dataset/scientific/scientific.valid.jsonl`
- `dataset/scientific/scientific.test.jsonl`
- `dataset/scientific/scientific_emb_256.npy`
- `dataset/scientific/256-512-256-128.rqvae.pth`

The same layout applies to `game` and `instrument`.

## Training

Scientific:

```bash
bash run.sh
```

Game:

```bash
bash run_game.sh
```

Manual launch example:

```bash
accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/scientific.yaml \
  --lr_rec=0.005 \
  --lr_id=0.0001 \
  --cycle=2
```

RQ-VAE pretraining:

```bash
cd RQVAE
bash run_pretrain.sh
```

## Baselines And Analysis

Scientific BPR-MF suite:

```bash
python baselines/bprmf/src/run_scientific_baseline_suite.py
```

Single Scientific BPR-MF comparison run:

```bash
python baselines/bprmf/src/compare_scientific_searec.py
```

Data and codebook analysis:

```bash
python scripts/analyze_recommendation_data.py
python scripts/analyze_codebook_utilization.py --config ./config/scientific.yaml --tag before
python scripts/plot_codebook_utilization.py
```

Notebook export helpers:

```bash
python scripts/export_notebook_outputs.py notebooks/SEARec_Data_Analysis.ipynb --output-dir notebooks/exports/SEARec_Data_Analysis_exports
python scripts/convert_ipynb_to_rmd.py notebooks/SEARec_Data_Analysis.ipynb --output notebooks/exports/SEARec_Data_Analysis_full.Rmd
```

## Current Repo Status

The report covers Game, Instrument, and Scientific. The runnable configs shipped in this repo currently include `scientific` and `game`. Instrument metrics and analysis artifacts are preserved in the report and outputs, but an `config/instrument.yaml` rerun configuration is not currently included.
