#!/usr/bin/env python
"""
Parse ETEGRec / SEA-Rec training logs into tidy result tables.

For each log it extracts:
  - the FINAL "Test Results: {...}" (post-finetune, the headline test metrics)
  - the "Best Validation Score"
  - the best per-epoch validation row (max ndcg@10)
  - run metadata inferred from the "Config: {...}" line (dataset, the four
    alignment-loss weights, use_features) and from the filename (variant, run_type)

Outputs:
  outputs/minh_ablation_results.csv
  outputs/minh_ablation_results.md
and (if scientific full run found) outputs/results_scientific.json + results_summary.csv

Usage:
  python scripts/parse_training_results.py                 # scan logs/
  python scripts/parse_training_results.py logs/a.log ...  # specific files
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
OUT_DIR = os.path.join(ROOT, "outputs")

METRIC_KEYS = ["recall@1", "recall@5", "ndcg@5", "recall@10", "ndcg@10"]
# np.float64(...) wrapper tolerant float parser
NUMRE = r"(?:np\.float64\()?([0-9.eE+-]+)\)?"


def parse_metric_dict(text):
    """Parse a dict-like '{'recall@1': np.float64(0.0074), ...}' into floats."""
    out = {}
    for k in ["recall@1", "recall@5", "ndcg@5", "recall@10", "ndcg@10", "ndcg@1"]:
        m = re.search(rf"'{re.escape(k)}':\s*{NUMRE}", text)
        if m:
            out[k] = float(m.group(1))
    return out


def infer_variant(fname):
    base = os.path.basename(fname).lower()
    table = [
        ("full_short", "Full SEA-Rec (short)"),
        ("no_kl", "w/o KL loss"),
        ("no_deccl", "w/o Decoder-CL"),
        ("no_align", "w/o Alignment (KL+DecCL)"),
        ("no_features", "w/o Feature Adapter"),
    ]
    for key, label in table:
        if key in base:
            return key, label
    if "etegrec_scientific" in base or "etegrec_game" in base:
        return "full", "Full SEA-Rec (full schedule)"
    return "unknown", base


def parse_log(path):
    with open(path, errors="ignore") as f:
        raw = f.read()
    text = raw.replace("\r", "\n")

    # config / metadata
    dataset = None
    cfg = {}
    cm = re.search(r"Config:\s*\{(.+?)\}\s*$", text, re.M)
    mds = re.search(r"'dataset':\s*'([^']+)'", text)
    if mds:
        dataset = mds.group(1)
    for key in ["rec_kl_loss", "id_kl_loss", "rec_dec_cl_loss", "id_dec_cl_loss",
                "use_features", "finetune_epochs", "epochs", "warm_epoch"]:
        m = re.search(rf"'{key}':\s*([^,}}]+)", text)
        if m:
            cfg[key] = m.group(1).strip()

    # final test results (last occurrence of a NON-"Pre" Test Results line)
    final_test = None
    best_val = None
    for m in re.finditer(r"(Pre )?Test Results:\s*(\{[^}]*\})", text):
        if not m.group(1):  # not "Pre"
            final_test = parse_metric_dict(m.group(2))
    bm = re.findall(r"(?<!Pre )Best Validation Score:\s*" + NUMRE, text)
    if bm:
        best_val = float(bm[-1])

    # best per-epoch validation (max ndcg@10) — works even if run not finished
    best_val_row = None
    val_rows = []
    for m in re.finditer(r"\[Epoch (\d+)\] Val Results:\s*(\{[^}]*\})", text):
        d = parse_metric_dict(m.group(2))
        d["epoch"] = int(m.group(1))
        val_rows.append(d)
    if val_rows:
        best_val_row = max(val_rows, key=lambda r: r.get("ndcg@10", -1))

    finished = final_test is not None
    return {
        "log": os.path.relpath(path, ROOT),
        "dataset": dataset or "?",
        "config": cfg,
        "finished": finished,
        "final_test": final_test,
        "best_val_score": best_val,
        "best_val_row": best_val_row,
        "n_val_evals": len(val_rows),
    }


def run_type_for(fname):
    base = os.path.basename(fname).lower()
    if base.startswith("smoke"):
        return "smoke"
    if "ablation_" in base:
        return "short"
    return "full"


def main():
    files = sys.argv[1:]
    if not files:
        files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    rows = []
    for path in files:
        if not os.path.exists(path):
            continue
        base = os.path.basename(path).lower()
        # skip non-result logs: the sweep driver wrapper (duplicates a variant log) and smoke tests
        if "driver" in base or base.startswith("smoke"):
            continue
        info = parse_log(path)
        if info["dataset"] == "?" and info["n_val_evals"] == 0:
            continue  # not a training log
        variant_key, variant_label = infer_variant(path)
        rt = run_type_for(path)
        test = info["final_test"]
        bvr = info["best_val_row"]
        # prefer final test; else fall back to best-val row (mark pending)
        src = test if test else (bvr if bvr else {})
        row = {
            "dataset": info["dataset"],
            "variant": variant_label,
            "variant_key": variant_key,
            "run_type": rt,
            "status": "DONE" if info["finished"] else ("RUNNING/partial" if info["n_val_evals"] else "PENDING"),
            "metric_source": "final_test" if test else ("best_val(partial)" if bvr else "none"),
            "best_epoch": (bvr or {}).get("epoch", ""),
            "n_val_evals": info["n_val_evals"],
            "best_val_ndcg@10": info["best_val_score"] if info["best_val_score"] is not None
                                else (bvr or {}).get("ndcg@10", ""),
        }
        for k in METRIC_KEYS:
            row[k] = round(src[k], 6) if k in src else ""
        row["log"] = info["log"]
        rows.append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    # de-dup: keep latest log per (dataset, variant_key, run_type)
    rows.sort(key=lambda r: r["log"])
    dedup = {}
    for r in rows:
        dedup[(r["dataset"], r["variant_key"], r["run_type"])] = r
    rows = list(dedup.values())
    # order
    order = {"full": 0, "full_short": 1, "no_kl": 2, "no_deccl": 3, "no_align": 4, "no_features": 5, "unknown": 9}
    rows.sort(key=lambda r: (r["dataset"], order.get(r["variant_key"], 9)))

    csv_path = os.path.join(OUT_DIR, "minh_ablation_results.csv")
    fields = ["dataset", "variant", "variant_key", "run_type", "status", "metric_source",
              "best_epoch", "n_val_evals", "best_val_ndcg@10"] + METRIC_KEYS + ["log"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown
    md = ["# Minh — parsed training/ablation results\n",
          "_Auto-generated by `scripts/parse_training_results.py` from `logs/`. "
          "`metric_source=final_test` = completed run (headline test metrics); "
          "`best_val(partial)` = run still in progress, value is best validation so far, NOT final test._\n",
          "| Dataset | Variant | Run type | Status | R@1 | R@5 | R@10 | N@5 | N@10 | Best epoch | Source | Log |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        def g(k):
            v = r.get(k, "")
            return f"{v:.4f}" if isinstance(v, float) else (str(v) if v != "" else "—")
        md.append(f"| {r['dataset']} | {r['variant']} | {r['run_type']} | {r['status']} | "
                  f"{g('recall@1')} | {g('recall@5')} | {g('recall@10')} | {g('ndcg@5')} | {g('ndcg@10')} | "
                  f"{r['best_epoch'] or '—'} | {r['metric_source']} | `{os.path.basename(r['log'])}` |")
    md_path = os.path.join(OUT_DIR, "minh_ablation_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    # scientific full headline json/csv
    full = next((r for r in rows if r["dataset"] == "scientific" and r["variant_key"] == "full"
                 and r["status"] == "DONE"), None)
    if full:
        with open(os.path.join(OUT_DIR, "results_scientific.json"), "w") as f:
            json.dump(full, f, indent=2)
        with open(os.path.join(OUT_DIR, "results_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow(full)

    print(f"[ok] {csv_path}")
    print(f"[ok] {md_path}")
    print(f"parsed {len(rows)} run rows:")
    for r in rows:
        print(f"  {r['dataset']:11s} {r['variant']:28s} {r['run_type']:6s} {r['status']:14s} "
              f"R@10={r.get('recall@10','—')} N@10={r.get('ndcg@10','—')}")


if __name__ == "__main__":
    main()
