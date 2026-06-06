# Why Recall@10 / NDCG@10 Look "Low" (Minh)

_All counts are from the local data (`outputs/data_analysis_*.json`); all metrics are from the completed Scientific run (`logs/etegrec_scientific_20260604_192028.log`)._

## The headline numbers
Completed **Full SEA-Rec** run on **Scientific** (best epoch 62, test set, full-catalog ranking):

| metric | value |
|---|---:|
| Recall@1 | 0.0075 |
| Recall@5 | 0.0265 |
| Recall@10 | **0.0419** |
| NDCG@5 | 0.0169 |
| NDCG@10 | **0.0219** |

At first glance Recall@10 ≈ 0.04 looks tiny. It is **expected and reasonable** for this task. Here is why, with numbers.

## 1. This is full-catalog ranking, not 1-of-100
The model does **not** pick the right item out of ~100 sampled negatives. The T5 decoder
beam-searches over the entire RQ-code space and the true item must rank in the **top 10 out
of the whole catalog**:
- Scientific: **25,848 items**
- Game: 25,612 items · Instrument: 24,587 items

A random ranker would get Recall@10 ≈ 10 / 25,848 ≈ **0.00039**. The model's **0.0419 is ~108×
random** — it is finding real structure; the absolute number is just small because the
denominator (the catalog) is huge.

## 2. Recall@10 is strict exact-match
Leave-one-out holds out **exactly one** correct next item per user. If that exact item is not
in the top-10 list, the user scores 0 — a *very similar* item earns **no partial credit**.
So Recall@10 = 0.0419 literally means **~4.2% of users get their one true held-out item into
the top 10** under exact match. NDCG@10 = 0.0219 further discounts by rank position.

## 3. The data is sparse and short-historied
From the local Scientific data:
- **Density = 0.0311%** (409,394 interactions in a 50,985 × 25,848 user–item matrix) — i.e.
  **~99.97% of the matrix is empty**.
- **Median user history = 5 items** (mean ≈ 7.0). Many users give the model only a handful of
  signals to predict the next item from.

(Game density 0.0330%, Instrument 0.0358% — all similarly sparse.)

## 4. The item distribution is long-tailed
- Scientific: **top 1% of items account for 13.8% of interactions; Gini = 0.49.**
- Game: top 1% = 18.7%, **Gini = 0.64**; Instrument: top 1% = 18.2%, Gini = 0.57.

A small head of popular items dominates; the long tail of items has very few interactions and
is genuinely hard to retrieve. Held-out targets in the tail are nearly impossible to place in
a top-10 drawn from 25k candidates.

## 5. Generative retrieval adds its own difficulty
The model must **generate a valid code path** `(c1, c2, c3, c4)` for the right item. Beam
search (`num_beams=20`) can miss the exact path even when the item is "close" in embedding
space — an extra source of strictness compared to a dot-product scorer.

## 6. So ~0.03–0.04 Recall@10 is reasonable
Given full-catalog exact-match ranking over ~26k items, ~0.03–0.04 Recall@10 is in the
**normal range** for generative recommendation on sparse Amazon-category data, not a sign the
model is broken.

## 7. Read it as *relative* improvement, not absolute %
The within-run evidence: the cyclic-aligned + finetuned model improves test metrics over the
pre-finetune checkpoint by a wide margin on the **same** strict setting:

| stage | Recall@10 | NDCG@10 |
|---|---:|---:|
| after pretrain (pre-finetune) | 0.0325 | 0.0164 |
| **final (Full SEA-Rec)** | **0.0419** | **0.0219** |
| relative gain | **+29%** | **+33%** |

The ablation study (removing KL / decoder-CL / feature adapter) is the apples-to-apples way to
read SEA-Rec's contribution — see the ablation table.

## Presentation paragraph (Vietnamese)
> **Recall@10 nhìn thấp vì đây không phải bài toán chọn 1 item trong 100 negatives.** Đây là
> full-ranking trên toàn bộ catalog hàng chục nghìn items (Scientific có 25,848 items). Model
> phải đưa **đúng** item thật (held-out) vào top 10, không có điểm cho item "gần giống". Vì vậy
> Recall@10 = 0.0419 nghĩa là khoảng **4.2% users** có đúng item held-out trong top 10 — và con
> số này đã **cao gấp ~108 lần** so với đoán ngẫu nhiên (≈0.0004). Dữ liệu lại rất thưa
> (mật độ 0.03%, lịch sử trung vị chỉ 5 item) và phân phối item lệch đuôi dài (Gini ≈ 0.49),
> nên giá trị tuyệt đối nhỏ là **bình thường**. Điều cần nhấn mạnh là **mức cải thiện tương đối**
> của SEA-Rec so với baseline, chứ không phải con số phần trăm tuyệt đối.
