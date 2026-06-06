# Why a Generative Method like ETEGRec / SEA-Rec is Needed (Minh)

_Grounded in the local data (`outputs/data_analysis_*.json`) and this repo's architecture._

## The problem, stated plainly
We have, per user, a short time-ordered list of items they interacted with, and we must
predict the **single next item**. The catalog is large and the data is sparse:

| dataset | users | items | density | median history | Gini (item popularity) |
|---|---:|---:|---:|---:|---:|
| scientific | 50,985 | 25,848 | 0.0311% | 5 | 0.49 |
| game | 94,762 | 25,612 | 0.0330% | 5 | 0.64 |
| instrument | 57,439 | 24,587 | 0.0358% | 6 | 0.57 |

So: **tens of thousands of candidate items**, **~99.97% of the user×item matrix empty**, and a
**long-tail** distribution where a few items dominate.

## 1. Why traditional ID-scoring recommenders struggle here
Classical recommenders learn one embedding **per item ID** and score user-vs-item by dot
product over the whole catalog. With 26k sparse, long-tailed items this means: a huge softmax,
poorly-learned tail-item embeddings (few interactions each), and no sharing of statistical
strength between related items. Accuracy on the tail collapses.

## 2. Generative recommendation reframes the task
Instead of scoring 26k IDs, **generative retrieval** turns recommendation into **sequence
generation**: predict the *next item's identifier token-by-token*. This is the ETEGRec idea.

## 3. RQ-VAE tokenizes items into compact semantic IDs
A Residual-Quantized VAE compresses each item's 256-d content embedding into a short
**discrete code sequence** (here 3 levels × 256 codes → an ID like `(35, 12, 240)`). Similar
items share **prefix codes**, so statistical strength **is** shared across related items —
directly attacking the sparsity/long-tail problem. (Measured: ~100% codebook utilization,
collision ≈1.5%; see the codebook section.)

## 4. T5 generates the next item's code sequence
A T5 encoder–decoder reads the user's history (as code sequences) and **generates** the code
sequence of the next item via beam search over the code space — never enumerating 26k IDs.

## 5. So why is plain ETEGRec not enough → SEA-Rec
There is an **objective mismatch**:
- The **RQ-VAE tokenizer** is trained to **reconstruct embeddings** (a compression objective).
- The **recommender** needs codes that are **predictable / useful for next-item generation**.

Codes that are great for reconstruction are not necessarily the codes a sequence model can
generate well. This is the **alignment gap** SEA-Rec targets.

## 6. What SEA-Rec adds (the components Minh ablates)
- **Cyclic co-training** between the tokenizer (M_id) and recommender (M_rec) so both adapt to each other instead of the tokenizer being frozen.
- **KL alignment loss** — aligns the two models' distributions over codes.
- **Decoder-level contrastive loss** — sharpens the decoder representations used to generate codes.
- **Feature adapter** — injects item content features into the T5 encoder stream.

The codebook analysis shows the co-trained tokenizer **specializes** its codes during training
(entropy drops, no collapse) — evidence the alignment objective actually reshapes the tokenizer
toward the recommendation task. The **ablation study** quantifies how much each component
contributes (see ablation table).

## 7. Concrete payoff on local data
On Scientific, end-to-end aligned training + finetuning lifts test Recall@10 from 0.0325 to
**0.0419 (+29%)** and NDCG@10 from 0.0164 to **0.0219 (+33%)** over the pre-finetune checkpoint
— on a strict full-catalog 25,848-item ranking where random ≈ 0.0004. A method this involved is
justified precisely because the **large catalog + extreme sparsity + long tail + exact
next-item** setting defeats simpler ID-scoring approaches.

## Paragraph (Vietnamese)
> Với catalog hàng chục nghìn item, dữ liệu cực thưa (~99.97% ô trống) và phân phối lệch đuôi
> dài, cách làm truyền thống (học 1 embedding cho mỗi item ID rồi chấm điểm trên toàn catalog)
> bị yếu ở phần đuôi vì mỗi item đuôi có quá ít tương tác. **Generative recommendation** biến
> bài toán thành **sinh chuỗi**: RQ-VAE mã hoá mỗi item thành một chuỗi mã rời rạc ngắn
> (semantic ID), item giống nhau chia sẻ tiền tố mã nên chia sẻ được sức mạnh thống kê; T5 học
> **sinh** chuỗi mã của item kế tiếp thay vì duyệt 26k ID. **ETEGRec** cung cấp khung
> generative đó. **SEA-Rec** cần thiết vì mục tiêu của tokenizer (tái tạo embedding) **lệch** với
> mục tiêu của recommender (mã dễ sinh cho next-item); SEA-Rec vá khoảng lệch này bằng
> **co-training xoay vòng + KL alignment + decoder contrastive + feature adapter**. Trên dữ liệu
> local, alignment + finetune nâng Recall@10 từ 0.0325 lên 0.0419 (+29%), NDCG@10 +33%.
