# Minh — Results Table (exact metrics from parsed logs only)

_Headline = completed full-schedule run. Short rows = controlled ablation sweep (same reduced
budget, comparable only to each other). PENDING rows have no completed log yet — **do not paste
numbers for them until their log finishes**. Regenerate with `python scripts/parse_training_results.py`._

## Headline — Full SEA-Rec, full schedule (test set, full-catalog ranking)

| Dataset | Variant | Run type | R@1 | R@5 | R@10 | N@1 | N@5 | N@10 | Log | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Scientific | Full SEA-Rec | full | 0.0075 | 0.0265 | 0.0419 | 0.0075 | 0.0169 | 0.0219 | `etegrec_scientific_20260604_192028.log` | best epoch 62; N@1=R@1 (single positive) |
| Game | Full SEA-Rec | full | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | — | **pending additional dataset experiment** — data + `config/game.yaml` ready, no completed log yet |
| Instrument | Full SEA-Rec | full | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | — | **pending additional dataset experiment** — data ready, but **`config/instrument.yaml` missing** |

> Game and Instrument are **pending**, not optional: `logs/` contains no Game run, and `config/`
> has no `instrument.yaml`. Plan: `outputs/report_sections/pending_game_instrument_plan.md`.

## Ablation sweep — Scientific (run_type = short; comparable to each other only) — COMPLETE

| Dataset | Variant | Run type | R@1 | R@5 | R@10 | N@5 | N@10 | Best epoch | Log | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Scientific | A. Full SEA-Rec (short) | short | 0.0074 | 0.0249 | 0.0399 | 0.0161 | 0.0210 | 13 | `ablation_scientific_full_short_20260605_173533.log` | short baseline for the sweep |
| Scientific | B. w/o KL loss | short | 0.0072 | 0.0248 | 0.0399 | 0.0160 | 0.0209 | 13 | `ablation_scientific_no_kl_20260605_192543.log` | ≈ baseline (within short-run noise) |
| Scientific | C. w/o Decoder-CL | short | 0.0072 | 0.0247 | 0.0398 | 0.0159 | 0.0208 | 13 | `ablation_scientific_no_deccl_20260605_211448.log` | ≈ baseline |
| Scientific | D. w/o Alignment (KL+DecCL) | short | 0.0071 | 0.0249 | 0.0402 | 0.0161 | 0.0210 | 13 | `ablation_scientific_no_align_20260605_225456.log` | ≈ baseline |
| Scientific | E. w/o Feature Adapter | short | 0.0074 | 0.0256 | 0.0408 | 0.0165 | 0.0213 | 11 | `ablation_scientific_no_features_20260606_003953.log` | **slightly above baseline — see anomaly note** |

> **Feature Adapter anomaly (inconclusive):** w/o Feature Adapter (R@10 0.0408 / N@10 0.0213)
> edges out the Full-short baseline (0.0399 / 0.0210) by ~0.0009 R@10. This is **not** evidence
> the adapter hurts: the gap is within **short-run / single-seed variance**, the adapter's
> **extra parameters** are under-trained at a 13-epoch budget, and the **SASRec collaborative
> embeddings are already strong**. Needs a full-schedule, multi-seed check before any conclusion.

## Within-run evidence (Scientific full schedule)
| stage | R@10 | N@10 |
|---|---:|---:|
| after pretrain (pre-finetune) | 0.0325 | 0.0164 |
| final (Full SEA-Rec) | **0.0419** | **0.0219** |
| relative gain from finetune | +29% | +33% |

> The live machine-parsed version of these tables is in `outputs/minh_ablation_results.md` and
> `outputs/minh_ablation_results.csv`. As each ablation log completes, re-running
> `python scripts/parse_training_results.py` fills the PENDING rows automatically.
