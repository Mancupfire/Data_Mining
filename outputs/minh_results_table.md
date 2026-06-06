# Minh — Results Table (exact metrics from parsed logs only)

_Headline = completed full-schedule run. Short rows = controlled ablation sweep (same reduced
budget, comparable only to each other). PENDING rows have no completed log yet — **do not paste
numbers for them until their log finishes**. Regenerate with `python scripts/parse_training_results.py`._

## Headline — Full SEA-Rec, full schedule (test set, full-catalog ranking)

| Dataset | Variant | Run type | R@1 | R@5 | R@10 | N@1 | N@5 | N@10 | Log | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Scientific | Full SEA-Rec | full | 0.0075 | 0.0265 | 0.0419 | 0.0075 | 0.0169 | 0.0219 | `etegrec_scientific_20260604_192028.log` | best epoch 62; N@1=R@1 (single positive) |
| Game | Full SEA-Rec | full | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | — | run after Scientific ablations finish |

## Ablation sweep — Scientific (run_type = short; comparable to each other only)

| Dataset | Variant | Run type | R@1 | R@5 | R@10 | N@5 | N@10 | Log | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Scientific | A. Full SEA-Rec (short) | short | RUNNING | RUNNING | RUNNING | RUNNING | RUNNING | `ablation_scientific_full_short_*.log` | short baseline for the sweep |
| Scientific | B. w/o KL loss | short | PENDING | PENDING | PENDING | PENDING | PENDING | `ablation_scientific_no_kl_*.log` | queued |
| Scientific | C. w/o Decoder-CL | short | PENDING | PENDING | PENDING | PENDING | PENDING | `ablation_scientific_no_deccl_*.log` | queued |
| Scientific | D. w/o Alignment (KL+DecCL) | short | PENDING | PENDING | PENDING | PENDING | PENDING | `ablation_scientific_no_align_*.log` | queued |
| Scientific | E. w/o Feature Adapter | short | PENDING | PENDING | PENDING | PENDING | PENDING | `ablation_scientific_no_features_*.log` | queued |

## Within-run evidence (Scientific full schedule)
| stage | R@10 | N@10 |
|---|---:|---:|
| after pretrain (pre-finetune) | 0.0325 | 0.0164 |
| final (Full SEA-Rec) | **0.0419** | **0.0219** |
| relative gain from finetune | +29% | +33% |

> The live machine-parsed version of these tables is in `outputs/minh_ablation_results.md` and
> `outputs/minh_ablation_results.csv`. As each ablation log completes, re-running
> `python scripts/parse_training_results.py` fills the PENDING rows automatically.
