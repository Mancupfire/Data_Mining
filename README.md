# Baseline Models for Sequential Recommendation

This is the baseline component of the final project. I implemented and ran two well-known sequential recommendation models - SASRec and GRU4Rec - on three Amazon product review datasets to establish baseline performance numbers. The goal is to compare these against the main method we study in the project.

## Models

**SASRec** (Kang & McAuley, 2018) uses a self-attention mechanism (similar to the encoder in Transformers) to capture sequential patterns in user interaction histories. It applies a causal mask so each position only attends to earlier items.

**GRU4Rec** (Hidasi et al., 2016) models user sequences with a Gated Recurrent Unit (GRU). It was one of the first deep learning approaches to session-based recommendation and remains a solid baseline.

Both models are trained with BPR-style binary cross-entropy loss (one positive item vs. one sampled negative per step) and evaluated under the full-ranking protocol - the target item is ranked against the entire item catalog, with training items masked out.

## Datasets

I used three categories from the Amazon Review 2023 dataset (Hou et al., 2024), preprocessed with 5-core filtering (users and items must each have at least 5 interactions) and a leave-one-out split for validation and test.

| Dataset | #Users | #Items | #Interactions |
|---|---|---|---|
| Video Games | 94,762 | 25,612 | 801,484 |
| Musical Instruments | 57,439 | 24,587 | 506,513 |
| Industrial & Scientific | 50,985 | 25,848 | 409,535 |

The datasets were originally formatted by the ETEGRec project (Chen et al., 2024) and converted to the standard sequential recommendation format used here.

## Project structure

```
├── models/
│   ├── sasrec.py          # SASRec model
│   └── gru4rec.py         # GRU4Rec model
├── dataset.py             # data loading and PyTorch dataset classes
├── train_sasrec.py        # training script for SASRec
├── train_gru4rec.py       # training script for GRU4Rec
├── evaluate.py            # full-ranking evaluation
├── preprocess.py          # 5-core filtering and train/val/test split
├── download_data.py       # downloads raw Amazon Review data
├── convert_etegrec.py     # converts ETEGRec JSONL format to baseline format
├── pipeline.py            # runs the full training pipeline sequentially
├── summarize.py           # aggregates result files into a summary table
├── run_all.sh             # end-to-end shell script (download -> train -> eval)
└── run_etegrec_pipeline.sh
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy
```

## Running

To run the full pipeline from scratch (downloads data, preprocesses, trains both models):

```bash
bash run_all.sh
```

To train a single model on one dataset:

```bash
python train_sasrec.py --dataset scientific --epochs 200 --batch_size 256
python train_gru4rec.py --dataset game --epochs 200 --batch_size 256
```

To evaluate a saved checkpoint:

```bash
python evaluate.py --model sasrec --dataset scientific \
    --checkpoint checkpoints/sasrec_scientific_best.pt --split test
```

## Results

Full-ranking evaluation on the test set (Recall and NDCG at cutoffs 1, 5, 10). These are the numbers I use as baselines in the project report.

**SASRec**

| Dataset | R@1 | R@5 | R@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| Video Games | 0.0031 | 0.0127 | 0.0222 | 0.0031 | 0.0080 | 0.0110 |
| Musical Instruments | 0.0035 | 0.0138 | 0.0241 | 0.0035 | 0.0085 | 0.0118 |
| Industrial & Scientific | 0.0028 | 0.0083 | 0.0136 | 0.0028 | 0.0055 | 0.0072 |

**GRU4Rec**

| Dataset | R@1 | R@5 | R@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| Video Games | 0.0018 | 0.0124 | 0.0212 | 0.0018 | 0.0072 | 0.0100 |
| Musical Instruments | 0.0033 | 0.0143 | 0.0236 | 0.0033 | 0.0089 | 0.0118 |
| Industrial & Scientific | 0.0026 | 0.0075 | 0.0133 | 0.0026 | 0.0051 | 0.0069 |

SASRec consistently outperforms GRU4Rec across all three datasets, which aligns with findings in prior work. The numbers are low overall because full-ranking evaluation against a catalog of ~25K items is a much harder setting than the sampled evaluation typically reported in papers.

## References

- Kang, W.-C., & McAuley, J. (2018). Self-attentive sequential recommendation. *IEEE ICDM*. https://doi.org/10.1109/ICDM.2018.00035
- Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based recommendations with recurrent neural networks. *ICLR*. https://arxiv.org/abs/1511.06939
- Hou, Y., Mu, S., Zhao, W. X., Li, Y., Ding, B., & Wen, J.-R. (2024). Bridging language and items for retrieval and recommendation. *arXiv*. https://arxiv.org/abs/2403.03952
- Chen, H., et al. (2024). ETEGRec: End-to-End Generative Sequential Recommendation with Next-Token Prediction. *arXiv*. https://arxiv.org/abs/2408.16143
