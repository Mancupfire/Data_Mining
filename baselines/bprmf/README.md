# BPR-MF Baseline

This directory vendors the BPR-MF code used for the SEA-Rec Scientific baseline comparison.

## What Was Kept

- core BPR-MF source under `src/`
- SEA-Rec-specific Scientific comparison scripts
- baseline package metadata and license

## What Was Removed

- the nested Git repository metadata
- bundled demo datasets that were unrelated to the SEA-Rec experiments
- local cache files

## Run

From the repository root:

```bash
python baselines/bprmf/src/compare_scientific_searec.py
python baselines/bprmf/src/run_scientific_baseline_suite.py
```

Both scripts now resolve `dataset/scientific/` and `outputs/` relative to the main repository root, so they no longer depend on the current working directory.
