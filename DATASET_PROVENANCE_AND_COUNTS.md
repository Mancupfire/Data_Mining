# DATASET PROVENANCE & COUNTS

Date: 2026-06-04. All numbers below are computed from local files in this repo — no external assumptions.

## 1. What datasets exist locally
Three datasets, each already **fully preprocessed** (leave-one-out splits + SASRec 256-d item embeddings + pretrained RQ-VAE). **There is NO `beauty` and NO `sports` data in this repo.**

| local folder | files |
|---|---|
| `dataset/scientific/` | `*.train/valid/test.jsonl`, `scientific.emb_map.json`, `scientific_emb_256.npy`, `256-512-256-128.rqvae.pth` (+ regenerated `item_features.npy`) |
| `dataset/game/` | same set, prefix `game` |
| `dataset/instrument/` | same set, prefix `instrument` |

## 2. Grounded counts (from local files)
- **Users** = lines in `*.test.jsonl` (leave-one-out → exactly one test target per user; train/valid/test all have the same user set).
- **Items** = `len(emb_map) − 1` (minus the `[PAD]` entry), cross-checked against the `*_emb_256.npy` row count.
- The split files contain **only `user_id`, `target_id`, `inter_history`** — **no timestamps, no ratings.**

| folder | users | items | train rows | valid rows | test rows |
|---|---|---|---|---|---|
| scientific | 50,985 | 25,848 | 259,992 | 50,985 | 50,985 |
| game | 94,762 | 25,612 | 530,300 | 94,762 | 94,762 |
| instrument | 57,439 | 24,587 | 339,519 | 57,439 | 57,439 |

(`items` matches the embedding matrix exactly: 25,848 / 25,612 / 24,587 rows.)

## 3. Source / version evidence (local only)
**Confirmed from files:**
- Item IDs are **Amazon ASINs** (`B0...`, e.g. `B098BR67DJ`) → Amazon Review data.
- Folder names map to Amazon Review categories: `scientific`=Industrial_and_Scientific, `game`=Video_Games, `instrument`=Musical_Instruments.
- **Likely Amazon Review *2023*:** 8–18% of items use `B0B/B0C/B0D` ASIN prefixes (2022–2023-era products) that are absent from the older Amazon 2014/2018 dumps. (scientific 18.1%, instrument 12.6%, game 8.5%.)
- `README.md` states embeddings/RQ-VAE/interactions were **downloaded from the ETEGRec paper's Google Drive** (`drive.google.com/.../1KiPpB7uq7eFc4qB74cFOxhtY3H8nWgAI`). So this data was obtained **pre-processed from the paper authors**, not produced by a local Amazon-2023 pipeline.

**Cannot be confirmed from files (no evidence present):**
- The exact Amazon Review **version string** (2023 is inferred from ASINs, not declared anywhere).
- The **download link/date/category filename** used originally — no metadata/README inside `dataset/`.
- **Raw counts before 5-core** — no raw review files exist locally, only the filtered splits.
- The **5-core filtering script** used to produce these splits.

## 4. Comparison with collaborator's counts
Collaborator's own 5-core on Amazon Review 2023:
- Beauty 22,363 u / 12,101 i · **Scientific 54,567 u / 27,229 i** · Sports 35,598 u / 18,357 i

Local: **scientific 50,985 u / 25,848 i**.
→ Same category, **same order of magnitude** (~51K vs ~55K users; ~25.8K vs ~27.2K items). The gap is consistent with a **different preprocessing variant** (e.g. iterative vs single-pass 5-core, min-history length, dedup, or a slightly different 2023 snapshot). They are plausibly the same Industrial_and_Scientific category, different filtering — **not** different datasets.
→ `game` (94,762 u) and `instrument` (57,439 u) do **not** correspond to Beauty or Sports. The earlier "Sports → game" mapping is a placeholder substitution, **not** a real category match.

## 5. The Milestone-1 "Scientific ~23K users / 2.1M items" claim
**"2.1M items" is almost certainly interactions/reviews, not unique items.**
- The local Scientific embedding matrix proves there are **25,848 unique items** — two orders of magnitude below 2.1M.
- No mid-size Amazon category has ~2M *unique products*; ~2M is the right scale for *reviews/interactions* (raw, before 5-core).
- The "~23K users" figure matches **neither** local (50,985) **nor** collaborator (54,567). I **cannot** reproduce 23K from any local file; it may be a different filtering threshold, a different snapshot, or a reporting error. **Unverifiable locally.**

## 6. Preprocessing protocol verification
- **5-core:** Cannot be re-derived from raw data (no raw files). Indirectly consistent — every user has ≥1 train + 1 valid + 1 test interaction. The exact ≥5 threshold is **not provable** from the filtered splits alone.
- **Leave-one-out:** Structurally confirmed — exactly **one valid and one test target per user**, train holds the earlier history (`data.py` builds `id_seq = history + [target]`). **Caveat:** the splits contain **no timestamps**, so I cannot prove the test item is the *chronologically latest* interaction; ordering is implicit in `inter_history` list order.
- **Full-ranking eval:** Generative — the T5 decoder beam-searches over the whole RQ-code space (`trainer.py:694/699`, `generate(..., n_return_sequences=10)`). **No sampled negatives.** Capped at top-10 returns.
- **Metrics @{1,5,10}:** `trainer.evaluate` returns `recall@{1,5,10}` and `ndcg@{1,5,10}` (`trainer.py:464-471`). All six exist. (The config `metrics:` string omits `ndcg@1`, but it is computed; `ndcg@1 == recall@1` for a single positive anyway.)

## 7. "Need from Minh" (to fully close provenance)
Local files are **not** enough to prove exact source/version. Please provide:
1. Original **download URL** + which **Amazon Review version** (confirm 2023) and the **category file name(s)** (e.g. `Industrial_and_Scientific.jsonl`).
2. Snapshot/version **date**.
3. **Raw counts before 5-core** (users / items / interactions) per category.
4. The **5-core filtering + leave-one-out script** used (single-pass vs iterative; min history length; how ties/timestamps were handled).
5. Confirmation of the Milestone-1 table semantics — i.e. that "2.1M" is the **interaction/review** count, and where the "23K users" figure came from.
