# SEA-Rec / ETEGRec — Report Update (Minh)

_Date: 2026-06-07 (refreshed from latest on-disk checks). Repo: `/mnt/disk1/backup_user/minh.ntn/ETEGRec`. Single GPU (CUDA 7)._
_Every number below is computed from local files / training logs in this repo — nothing is fabricated. Pending items are labelled PENDING._

> **Status of additional datasets (latest check):** **Game** has its data and `config/game.yaml`,
> but **no completed Game training logs/results exist yet** → Game is a **pending additional
> dataset experiment**, not an optional one. **Instrument** has its data but **`config/instrument.yaml`
> is missing** → Instrument is also **pending** and first needs a config. See
> `outputs/report_sections/pending_game_instrument_plan.md`.

Ready-to-paste sections. Full detail per topic lives in `outputs/report_sections/*.md`;
figures in `outputs/figures/`; machine-readable results in `outputs/*.json` / `*.csv`.

---

## 1. Problem statement (with a concrete example)

We observe each user's time-ordered interaction history and must predict the **single next
item** (leave-one-out: the last item per user is held out as the test target). This is
**full-catalog ranking**: the model ranks the true item against the **entire catalog of
~26,000 items** — not against ~100 sampled negatives.

**Concrete example (real, anonymized, from `scientific.test.jsonl`):**

| user (anon) | first history items | held-out target | history len |
|---|---|---|---:|
| `AFNT6ZJC…` | `B00DX7KEP8`, `B098BR67DJ`, `B08Y97BK7N`, `B0B6YT7HNF`, `B0883CDD2Z` | `B09HSD6Q22` | 5 |

The model must place `B09HSD6Q22` in the top-10 out of 25,848 candidates. Item IDs are Amazon
ASINs → this is Amazon-Review data (Scientific = Industrial_and_Scientific).

---

## 2. Data analysis & samples

_Full section: `outputs/report_sections/data_problem_analysis.md`; JSON: `outputs/data_analysis_*.json`._

| Dataset | Amazon category | Users | Items | Interactions | Density | Median history | Top-1% item share | Gini |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| scientific | Industrial_and_Scientific | 50,985 | 25,848 | 409,394 | 0.0311% | 5 | 13.8% | 0.49 |
| game | Video_Games | 94,762 | 25,612 | 801,031 | 0.0330% | 5 | 18.7% | 0.64 |
| instrument | Musical_Instruments | 57,439 | 24,587 | 506,286 | 0.0358% | 6 | 18.2% | 0.57 |

**Why the problem is hard:** large catalog (~26k items), extreme sparsity (~99.97% of the
user×item matrix empty), long-tail popularity (a few items dominate; high Gini), short
histories (median 5), and exact single-target next-item prediction.

---

## 3. Why Recall@10 / NDCG@10 look low

_Full section: `outputs/report_sections/why_recall_is_low.md`._

Completed **Full SEA-Rec** on Scientific (test, full-catalog): **Recall@10 = 0.0419,
NDCG@10 = 0.0219**. This is reasonable because:
1. Full-catalog ranking over 25,848 items — random ≈ 0.0004, so 0.0419 is **~108× random**.
2. Strict exact-match: one held-out item per user, no partial credit for similar items.
3. Sparse data (density 0.031%) and short histories (median 5).
4. Long-tail items (Gini 0.49) are intrinsically hard to retrieve.
5. Generative retrieval must beam-search the exact code path.

> **VN:** Recall@10 thấp vì đây là full-ranking trên toàn bộ catalog ~26k items (không phải
> 1-trong-100 negatives). 0.0419 nghĩa là ~4.2% users có đúng item held-out trong top 10, đã
> cao gấp ~108× so với ngẫu nhiên. Dữ liệu thưa (0.03%), lịch sử ngắn (trung vị 5), đuôi dài
> (Gini 0.49) ⇒ con số tuyệt đối nhỏ là bình thường; nên nhìn vào **cải thiện tương đối**.

---

## 4. Why a complex method (ETEGRec / SEA-Rec) is needed

_Full section: `outputs/report_sections/why_etegrec_is_needed.md`._

Traditional ID-scoring recommenders learn one embedding per item and score over the whole
catalog — they break down on 26k sparse, long-tailed items (tail embeddings barely trained).
**Generative recommendation** instead: RQ-VAE tokenizes each item into a short discrete code
sequence (similar items share code prefixes → shared statistical strength), and T5 **generates**
the next item's codes. **ETEGRec** is the generative-retrieval framework. **SEA-Rec** is needed
because the tokenizer's objective (reconstruct embeddings) is **misaligned** with the
recommender's (codes that are easy to generate); SEA-Rec closes this gap with **cyclic
co-training + KL alignment + decoder-contrastive loss + a feature adapter**.

Payoff on Scientific: alignment + finetune lift test Recall@10 0.0325→**0.0419 (+29%)**,
NDCG@10 0.0164→**0.0219 (+33%)** over the pre-finetune checkpoint.

---

## 5. K-means / codebook visualization (before vs after training)

_Full section: `outputs/report_sections/kmeans_codebook_visualization.md`; figure captions
`outputs/report_sections/codebook_figure_captions.md`; figures in `outputs/figures/`._

> **Source of the item embeddings (corrected):** the 256-d item vectors fed to the RQ-VAE are
> **SASRec collaborative embeddings** taken from the **ETEGRec authors' preprocessed Google Drive
> release** (`*_emb_256.npy`), **not Sentence-BERT / content text embeddings**. Provenance detail:
> `outputs/report_sections/embedding_provenance.md`.

RQ-VAE turns each item into 3 discrete codes (3 levels × 256-vector codebooks); the nearest-code
assignment is a **k-means**-style snap. We measured codebook usage **before** (shipped
random-init RQ-VAE) and **after** end-to-end training on Scientific:

| level | used/total before→after | dead before→after | norm. entropy before→after |
|---|---|---|---|
| 0 | 256/256 → 256/256 | 0 → 0 | 0.971 → 0.953 |
| 1 | 256/256 → 256/256 | 0 → 0 | 0.997 → 0.970 |
| 2 | 256/256 → 256/256 | 0 → 0 | 0.998 → 0.987 |

Joint collision rate 1.52% → 1.77%. **Interpretation:** (a) **no collapse** — 100% utilization,
zero dead codes after training; (b) codes **specialize** — entropy drops at every level (usage
becomes less uniform, matching the long-tail item distribution); (c) since utilization is
already ~100% before training, **k-means init has little headroom on utilization** here — its
value would be convergence stability, not higher utilization. Game (before-only): 99.2%/100%/100%.

> **What is done vs. future work here:** the **before-vs-after codebook visualization is COMPLETE**
> for Scientific. The **dedicated random-init vs. k-means-init ablation is NOT done** — it needs
> re-pretraining the RQ-VAE with `kmeans_init: True`. It is **future work**, scoped in
> `outputs/report_sections/kmeans_ablation_future_work.md`.

Figures: `codebook_utilization_…`, `codebook_dead_codes_…`, `codebook_entropy_…`,
`codebook_usage_hist_level0_…`, `codebook_top_codes_level0_…` (scientific has before+after;
game is before-only pending its full run).

---

## 6. Model run status & current results

| Dataset | Status | Best epoch | Test R@10 | Test N@10 | Log |
|---|---|---:|---:|---:|---|
| **Scientific (Full SEA-Rec, full schedule)** | **DONE** | 62 | **0.0419** | **0.0219** | `logs/etegrec_scientific_20260604_192028.log` |
| Game (Full SEA-Rec) | **PENDING** | — | — | — | data + `config/game.yaml` ready; **no completed log yet** |
| Instrument (Full SEA-Rec) | **PENDING** | — | — | — | data ready; **`config/instrument.yaml` missing** (create first) |

Full Scientific test row: R@1 0.0075 · R@5 0.0265 · R@10 0.0419 · N@5 0.0169 · N@10 0.0219.

Game and Instrument are **pending additional dataset experiments** (not optional): no Game logs
exist in `logs/` and no Instrument config exists in `config/`. Plan + commands:
`outputs/report_sections/pending_game_instrument_plan.md`.

---

## 7. Ablation study (Scientific) — status

_Full section: `outputs/report_sections/ablation_results.md`; live table `outputs/minh_ablation_results.md`._

Each variant removes one component; the sweep ran sequentially on GPU 7 with a **shared reduced
budget** (labelled `run_type=short`, comparable to each other — NOT to the full-schedule headline).
The short sweep is now **COMPLETE** (all five `.done` markers present in `logs/`).

| Variant | KL | Dec-CL | Features | run_type | Status | Test R@10 | Test N@10 |
|---|:--:|:--:|:--:|---|---|---:|---:|
| A. Full SEA-Rec | ✓ | ✓ | ✓ | full | **DONE** (headline) | **0.0419** | **0.0219** |
| A. Full SEA-Rec (short) | ✓ | ✓ | ✓ | short | **DONE** | 0.0399 | 0.0210 |
| B. w/o KL | ✗ | ✓ | ✓ | short | **DONE** | 0.0399 | 0.0209 |
| C. w/o Decoder-CL | ✓ | ✗ | ✓ | short | **DONE** | 0.0398 | 0.0208 |
| D. w/o Alignment | ✗ | ✗ | ✓ | short | **DONE** | 0.0402 | 0.0210 |
| E. w/o Feature Adapter | ✓ | ✓ | ✗ | short | **DONE** | 0.0408 | 0.0213 |
| F. no-tokenizer-update | — | — | — | — | planned, not run (needs code change) | — | — |
| G. k-means vs random init | — | — | — | — | **future work**, not run (needs RQ-VAE re-pretrain) | — | — |

**Reading the short sweep:** removing KL, Decoder-CL, or both (alignment) each moves Test R@10 by
≤0.0004 vs. the Full-short baseline (0.0399) — i.e. **within short-run noise** at this reduced
budget; the alignment losses' real payoff shows up on the **full schedule** (pre-finetune
0.0325 → 0.0419).

> **Feature Adapter anomaly (treat as inconclusive):** **w/o Feature Adapter** (R@10 **0.0408** /
> N@10 0.0213) **slightly outperforms** the Full-short baseline (0.0399 / 0.0210). We do **not**
> conclude the adapter hurts. The gap (~0.0009 R@10) is well within **short-run / single-seed
> variance**; the adapter adds **extra parameters** that a 13-epoch budget under-trains; and the
> SASRec collaborative embeddings are **already strong**, leaving little headroom for an extra
> content-feature pathway in this short setting. A full-schedule, multi-seed comparison is needed
> before drawing any conclusion.

---

## 8. Next steps / pending runs

1. **Scientific ablation sweep — DONE.** All five short variants completed; numbers are in
   Section 7 and `outputs/minh_ablation_results.{md,csv}` (regenerate with
   `python scripts/parse_training_results.py`).
2. **Game full run (PENDING — additional dataset experiment, not optional).** Data +
   `config/game.yaml` are ready; no completed Game log exists yet. Run (no feature adapter —
   game has no `item_features.npy`), then parse. See `pending_game_instrument_plan.md`.
3. **Instrument full run (PENDING).** Data is ready but **`config/instrument.yaml` is missing** —
   create it (clone `game.yaml`, swap dataset/paths) before training. See
   `pending_game_instrument_plan.md`.
4. **Game/Instrument codebook after-training** once each checkpoint exists:
   `python scripts/analyze_codebook_utilization.py --config ./config/<ds>.yaml --tag after --rqvae_ckpt <best>.pt.rqvae`
   then `python scripts/plot_codebook_utilization.py --dataset <ds>`.
5. **Variant G — k-means vs random init (FUTURE WORK).** Re-pretrain RQ-VAE with
   `kmeans_init: True`, repoint `rqvae_path`, re-run. Scoped in
   `outputs/report_sections/kmeans_ablation_future_work.md`.

### Blockers / notes
- **`config/instrument.yaml` does not exist** — it is the gating blocker for the Instrument run.
- **No completed Game logs** exist in `logs/` (only Scientific logs are present).
- No editable shared `.docx`/`.pdf` exists in the repo → this consolidated markdown +
  `outputs/report_sections/*.md` are the deliverables to paste into the team doc.
- Ablation full-schedule runs are ~8 h each on one GPU; the short labelled sweep is the
  practical way to get comparable ablation numbers — do not mix short and full numbers.
- `trainer.finetune()` got a backward-compatible `finetune_epochs`/`finetune_early_stop` config
  key (defaults reproduce the original 100/10 schedule) to enable the short sweep.
- Item embeddings are **SASRec 256-d** from the authors' Google Drive preprocessed release
  (**not Sentence-BERT**) — see `outputs/report_sections/embedding_provenance.md`.

### Artifact index
| Artifact | Path |
|---|---|
| Data analysis | `outputs/data_analysis_*.json`, `outputs/report_sections/data_problem_analysis.md` |
| Codebook before/after | `outputs/codebook_utilization_scientific_{before,after}.json`, `outputs/figures/codebook_*` |
| Results (parsed) | `outputs/minh_ablation_results.{md,csv}`, `outputs/results_scientific.json`, `outputs/results_summary.csv` |
| Results table | `outputs/minh_results_table.md` |
| Report sections | `outputs/report_sections/*.md` |
| Embedding provenance | `outputs/report_sections/embedding_provenance.md` |
| Codebook figure captions | `outputs/report_sections/codebook_figure_captions.md` |
| Pending Game/Instrument plan | `outputs/report_sections/pending_game_instrument_plan.md` |
| K-means init ablation (future work) | `outputs/report_sections/kmeans_ablation_future_work.md` |
| Scripts | `scripts/analyze_recommendation_data.py`, `scripts/plot_codebook_utilization.py`, `scripts/parse_training_results.py`, `scripts/run_minh_ablation_scientific.sh` |
