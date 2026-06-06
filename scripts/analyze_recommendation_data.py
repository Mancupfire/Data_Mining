#!/usr/bin/env python
"""
Analyze a preprocessed ETEGRec / SEA-Rec dataset and produce:
  - outputs/data_analysis_<dataset>.json      (machine-readable summary)
  - outputs/report_sections/data_problem_analysis.md  (human-readable, appended per dataset)

Data layout (local, leave-one-out):
  dataset/<name>/<name>.train.jsonl   augmented training rows (one per history prefix)
  dataset/<name>/<name>.valid.jsonl   one row per user
  dataset/<name>/<name>.test.jsonl    one row per user (FULL history -> held-out target)
  dataset/<name>/<name>.emb_map.json  {item_asin: id, ..., "[PAD]": 0}

Each jsonl row: {"user_id", "target_id", "inter_history": [...]}

The TEST row holds the longest observed history per user, so the full per-user
interaction sequence = inter_history + [target_id]. We use that for the
per-user length distribution, item-popularity and sparsity statistics.
"""
import argparse
import json
import os
import statistics
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset")
OUT_DIR = os.path.join(ROOT, "outputs")
REPORT_DIR = os.path.join(OUT_DIR, "report_sections")

# Amazon Review category each local folder corresponds to (grounded in DATASET_PROVENANCE).
CATEGORY = {
    "scientific": "Amazon Industrial_and_Scientific",
    "game": "Amazon Video_Games",
    "instrument": "Amazon Musical_Instruments",
}


def count_lines(path):
    n = 0
    with open(path) as f:
        for _ in f:
            n += 1
    return n


def percentile(sorted_vals, p):
    """Nearest-rank percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def gini(counts):
    """Gini coefficient of item interaction counts (0 = uniform, 1 = all on one item)."""
    xs = sorted(counts)
    n = len(xs)
    if n == 0:
        return 0.0
    cum = 0
    total = 0
    for i, x in enumerate(xs, 1):
        cum += i * x
        total += x
    if total == 0:
        return 0.0
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def analyze(dataset):
    ddir = os.path.join(DATA_DIR, dataset)
    train_p = os.path.join(ddir, f"{dataset}.train.jsonl")
    valid_p = os.path.join(ddir, f"{dataset}.valid.jsonl")
    test_p = os.path.join(ddir, f"{dataset}.test.jsonl")
    map_p = os.path.join(ddir, f"{dataset}.emb_map.json")
    for p in (train_p, valid_p, test_p, map_p):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    emb_map = json.load(open(map_p))
    n_items = len(emb_map) - 1  # minus [PAD]

    n_train_rows = count_lines(train_p)
    n_valid_users = count_lines(valid_p)
    n_test_users = count_lines(test_p)

    # Per-user full sequence comes from the test row (longest history + held-out target).
    hist_lens = []          # length of the OBSERVED history used to predict the test target
    seq_lens = []           # full interaction sequence length = history + target
    item_counter = Counter()
    samples = []
    total_interactions = 0

    with open(test_p) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            hist = r.get("inter_history", []) or []
            tgt = r.get("target_id")
            seq = list(hist) + ([tgt] if tgt is not None else [])
            hist_lens.append(len(hist))
            seq_lens.append(len(seq))
            total_interactions += len(seq)
            for it in seq:
                item_counter[it] += 1
            if idx < 5:
                samples.append({
                    "user_id": r.get("user_id", "")[:8] + "...",   # anonymized
                    "first_5_history": hist[:5],
                    "target_id": tgt,
                    "history_length": len(hist),
                })

    seq_sorted = sorted(seq_lens)
    n_users = n_test_users
    sparsity = total_interactions / float(n_users * n_items) if n_users and n_items else 0.0

    # Item popularity / long tail
    freqs = sorted(item_counter.values(), reverse=True)
    top10 = item_counter.most_common(10)
    n_top1pct = max(1, int(round(0.01 * len(freqs))))
    top1pct_share = sum(freqs[:n_top1pct]) / float(total_interactions) if total_interactions else 0.0
    g = gini(list(item_counter.values()))
    cold_items = n_items - len(item_counter)  # items never seen in any user sequence

    summary = {
        "dataset": dataset,
        "amazon_category": CATEGORY.get(dataset, "unknown"),
        "n_users": n_users,
        "n_items": n_items,
        "n_train_rows_augmented": n_train_rows,
        "n_valid_users": n_valid_users,
        "n_test_users": n_test_users,
        "total_interactions_full_sequences": total_interactions,
        "history_length": {
            "mean": round(statistics.mean(hist_lens), 4) if hist_lens else 0,
            "median": statistics.median(hist_lens) if hist_lens else 0,
            "min": min(hist_lens) if hist_lens else 0,
            "max": max(hist_lens) if hist_lens else 0,
        },
        "sequence_length_percentiles": {
            "p25": percentile(seq_sorted, 25),
            "p50": percentile(seq_sorted, 50),
            "p75": percentile(seq_sorted, 75),
            "p90": percentile(seq_sorted, 90),
            "p95": percentile(seq_sorted, 95),
            "p99": percentile(seq_sorted, 99),
        },
        "sparsity": {
            "interactions": total_interactions,
            "users_x_items": n_users * n_items,
            "density": round(sparsity, 8),
            "density_pct": round(sparsity * 100, 6),
        },
        "long_tail": {
            "top_10_items_id_count": top10,
            "top_1pct_items": n_top1pct,
            "top_1pct_interaction_share": round(top1pct_share, 4),
            "gini_coefficient": round(g, 4),
            "items_never_in_test_sequences": cold_items,
        },
        "concrete_examples": samples,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f"data_analysis_{dataset}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    return summary, out_json


def render_md(summaries):
    """Render a single markdown section covering all analyzed datasets."""
    lines = []
    lines.append("# Data & Problem Analysis (Minh)\n")
    lines.append("_Auto-generated by `scripts/analyze_recommendation_data.py`. All numbers are computed "
                 "directly from the local `dataset/<name>/*.jsonl` and `*.emb_map.json` files._\n")

    lines.append("## Problem statement for non-experts\n")
    lines.append(
        "- We observe each user's **historical interactions** (a time-ordered list of item IDs they engaged with).\n"
        "- **Task:** predict the **single next item** the user will interact with (leave-one-out: the last item of "
        "each user is held out as the test target).\n"
        "- **Crucially**, this is *not* the easy \"pick 1 out of ~100 sampled negatives\" setup. Evaluation is "
        "**full-ranking**: the model must rank the one correct item against the **entire catalog of tens of thousands "
        "of items**.\n"
        "- Therefore **Recall@10 is naturally low**: the model must place the exact held-out item inside the top 10 "
        "out of the whole catalog. Recall@10 = 0.04 means ~4% of users get their true next item into the top 10 under "
        "this strict exact-match, full-catalog setting.\n")

    # overview table
    lines.append("## Dataset overview\n")
    lines.append("| Dataset | Amazon category | Users | Items | Train rows (augmented) | Interactions | Density |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s['dataset']} | {s['amazon_category']} | {s['n_users']:,} | {s['n_items']:,} | "
            f"{s['n_train_rows_augmented']:,} | {s['total_interactions_full_sequences']:,} | "
            f"{s['sparsity']['density_pct']:.4f}% |")
    lines.append("")

    for s in summaries:
        d = s["dataset"]
        lines.append(f"## {d} ({s['amazon_category']})\n")
        h = s["history_length"]
        p = s["sequence_length_percentiles"]
        lines.append(f"- **Users:** {s['n_users']:,} &nbsp; **Unique items:** {s['n_items']:,}")
        lines.append(f"- **Train rows (augmented prefixes):** {s['n_train_rows_augmented']:,} "
                     f"&nbsp; **Valid/Test users:** {s['n_valid_users']:,} / {s['n_test_users']:,}")
        lines.append(f"- **Total interactions (full sequences):** {s['total_interactions_full_sequences']:,}")
        lines.append(f"- **Observed history length:** mean {h['mean']}, median {h['median']}, "
                     f"min {h['min']}, max {h['max']}")
        lines.append(f"- **Full-sequence length percentiles:** p25={p['p25']}, p50={p['p50']}, p75={p['p75']}, "
                     f"p90={p['p90']}, p95={p['p95']}, p99={p['p99']}")
        sp = s["sparsity"]
        lines.append(f"- **Sparsity:** {sp['interactions']:,} interactions / "
                     f"({s['n_users']:,} users × {s['n_items']:,} items) = **density {sp['density_pct']:.4f}%** "
                     f"(i.e. ~{100 - sp['density_pct']:.2f}% of the user×item matrix is empty).")
        lt = s["long_tail"]
        lines.append(f"- **Long tail:** top 1% of items ({lt['top_1pct_items']:,} items) account for "
                     f"**{lt['top_1pct_interaction_share'] * 100:.1f}%** of all interactions; "
                     f"**Gini = {lt['gini_coefficient']}** (0=uniform, 1=fully concentrated); "
                     f"{lt['items_never_in_test_sequences']:,} items never appear in any test sequence.")
        lines.append("\n**Top 10 most frequent items (item_id, interaction count):**\n")
        lines.append("| rank | item_id | count |")
        lines.append("|---:|---|---:|")
        for i, (it, c) in enumerate(lt["top_10_items_id_count"], 1):
            lines.append(f"| {i} | `{it}` | {c:,} |")
        lines.append("\n**Concrete anonymized examples (held-out test target):**\n")
        lines.append("| user (anon) | first 5 history items | target (held-out) | history len |")
        lines.append("|---|---|---|---:|")
        for ex in s["concrete_examples"]:
            hist = ", ".join(f"`{x}`" for x in ex["first_5_history"]) or "(empty)"
            lines.append(f"| {ex['user_id']} | {hist} | `{ex['target_id']}` | {ex['history_length']} |")
        lines.append("")

    lines.append("## Why the problem is hard\n")
    lines.append(
        "1. **Large item catalog** — tens of thousands of candidate items; the model ranks the true item against all of them.\n"
        "2. **Sparse interactions** — density is a tiny fraction of a percent; most user×item pairs are never observed.\n"
        "3. **Long-tail item distribution** — a small head of popular items dominates, leaving most items with very few "
        "interactions (high Gini), so tail items are hard to learn and to recommend.\n"
        "4. **Exact next-item prediction** — leave-one-out gives exactly one correct held-out item per user; similar "
        "items get no partial credit.\n"
        "5. **Short histories** — many users have very few interactions, giving the model little signal per user.\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="scientific | game | instrument")
    ap.add_argument("--render_only_this", action="store_true",
                    help="render md from only this dataset's json (default: merge all available json)")
    args = ap.parse_args()

    summary, out_json = analyze(args.dataset)
    print(f"[ok] wrote {out_json}")
    print(json.dumps({k: summary[k] for k in
                      ("dataset", "n_users", "n_items", "total_interactions_full_sequences")}, indent=2))

    # Re-render the combined markdown from whatever per-dataset jsons exist (so calling
    # the script per dataset accumulates a single tidy section).
    os.makedirs(REPORT_DIR, exist_ok=True)
    summaries = []
    for name in ("scientific", "game", "instrument"):
        jp = os.path.join(OUT_DIR, f"data_analysis_{name}.json")
        if os.path.exists(jp):
            summaries.append(json.load(open(jp)))
    md = render_md(summaries)
    md_path = os.path.join(REPORT_DIR, "data_problem_analysis.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"[ok] wrote {md_path} (covering {[s['dataset'] for s in summaries]})")


if __name__ == "__main__":
    main()
