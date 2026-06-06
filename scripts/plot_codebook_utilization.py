#!/usr/bin/env python
"""
Plot RQ-VAE codebook utilization BEFORE vs AFTER end-to-end training.

Reads:
  outputs/codebook_utilization_<dataset>_before.json
  outputs/codebook_utilization_<dataset>_after.json   (optional; if missing -> before-only)

Writes (matplotlib only) into outputs/figures/:
  codebook_utilization_<dataset>_before_after.png   used/total per level
  codebook_dead_codes_<dataset>_before_after.png    dead codes per level
  codebook_entropy_<dataset>_before_after.png        normalized entropy per level
  codebook_usage_hist_level0_<dataset>_before_after.png   level-0 assignment-count histogram
  codebook_top_codes_level0_<dataset>_before_after.png    top-10 used codes (optional)

The histogram needs the full per-code assignment counts, which are not stored in
the JSON, so it is recomputed by re-encoding the item embeddings through the
before/after RQ-VAE checkpoints (CPU by default, so it doesn't fight a running GPU job).
If recompute fails (e.g. checkpoint missing) the script still emits the JSON-based figures.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def load_json(dataset, tag):
    p = os.path.join(OUT_DIR, f"codebook_utilization_{dataset}_{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get(tag)


def levels_of(d):
    pl = d["per_level"]
    return sorted(pl.keys(), key=lambda k: int(k.split("_")[1]))


def grouped_bar(ax, levels, before_vals, after_vals, ylabel, title, after_avail):
    x = np.arange(len(levels))
    w = 0.38
    ax.bar(x - w / 2, before_vals, w, label="before training", color="#4C72B0")
    if after_avail:
        ax.bar(x + w / 2, after_vals, w, label="after training", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"level {i}" for i in range(len(levels))])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def fig_used(dataset, before, after, after_avail):
    levels = levels_of(before)
    bu = [before["per_level"][l]["used_codes"] for l in levels]
    au = [after["per_level"][l]["used_codes"] for l in levels] if after_avail else []
    tot = before["per_level"][levels[0]]["total_codes"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bar(ax, levels, bu, au,
                f"used codes (of {tot})",
                f"Codebook utilization — {dataset} (before vs after)\n"
                f"each level has {tot} codes", after_avail)
    ax.set_ylim(0, tot * 1.08)
    for i, v in enumerate(bu):
        ax.text(i - 0.19, v + tot * 0.01, str(v), ha="center", fontsize=8)
    if after_avail:
        for i, v in enumerate(au):
            ax.text(i + 0.19, v + tot * 0.01, str(v), ha="center", fontsize=8)
    out = os.path.join(FIG_DIR, f"codebook_utilization_{dataset}_before_after.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_dead(dataset, before, after, after_avail):
    levels = levels_of(before)
    bd = [before["per_level"][l]["dead_codes"] for l in levels]
    ad = [after["per_level"][l]["dead_codes"] for l in levels] if after_avail else []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bar(ax, levels, bd, ad, "dead (unused) codes",
                f"Dead codes per level — {dataset} (before vs after)", after_avail)
    out = os.path.join(FIG_DIR, f"codebook_dead_codes_{dataset}_before_after.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_entropy(dataset, before, after, after_avail):
    levels = levels_of(before)
    be = [before["per_level"][l]["normalized_entropy"] for l in levels]
    ae = [after["per_level"][l]["normalized_entropy"] for l in levels] if after_avail else []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bar(ax, levels, be, ae, "normalized entropy (1.0 = uniform usage)",
                f"Codebook usage entropy per level — {dataset} (before vs after)", after_avail)
    ax.set_ylim(0.85, 1.005)
    for i, v in enumerate(be):
        ax.text(i - 0.19, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    if after_avail:
        for i, v in enumerate(ae):
            ax.text(i + 0.19, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    out = os.path.join(FIG_DIR, f"codebook_entropy_{dataset}_before_after.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_top_codes(dataset, before, after, after_avail):
    """Top-10 most-used level-0 codes, before vs after."""
    bt = dict(before["per_level"]["level_0"]["top_10_most_used_codes"])
    at = dict(after["per_level"]["level_0"]["top_10_most_used_codes"]) if after_avail else {}
    codes = sorted(set(bt) | set(at), key=lambda c: -(bt.get(c, 0) + at.get(c, 0)))[:12]
    x = np.arange(len(codes)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, [bt.get(c, 0) for c in codes], w, label="before", color="#4C72B0")
    if after_avail:
        ax.bar(x + w / 2, [at.get(c, 0) for c in codes], w, label="after", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([str(c) for c in codes], rotation=45, fontsize=8)
    ax.set_xlabel("level-0 code id"); ax.set_ylabel("# items assigned")
    ax.set_title(f"Most-used level-0 codes — {dataset} (before vs after)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIG_DIR, f"codebook_top_codes_level0_{dataset}_before_after.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def recompute_level_counts(dataset, config_path, before_ckpt, after_ckpt, device="cpu"):
    """Re-encode item embeddings to get full per-code assignment counts (level 0)."""
    import torch  # noqa
    from analyze_codebook_utilization import (build_config, load_rqvae,
                                              load_embeddings, get_all_indices)
    config = build_config(config_path)
    emb = load_embeddings(config)
    n_codes = config["num_emb_list"][0]
    res = {}
    for tag, ckpt in (("before", before_ckpt), ("after", after_ckpt)):
        if ckpt is None or not os.path.exists(ckpt):
            continue
        model = load_rqvae(config, ckpt, device)
        idx = get_all_indices(model, emb, device)
        counts = np.bincount(idx[:, 0], minlength=n_codes)
        res[tag] = counts
    return res


def fig_hist_level0(dataset, counts):
    """Histogram of per-code assignment counts at level 0, before vs after."""
    if "before" not in counts:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    allvals = np.concatenate([v for v in counts.values()])
    bins = np.linspace(0, allvals.max(), 30)
    ax.hist(counts["before"], bins=bins, alpha=0.6, label="before training", color="#4C72B0")
    if "after" in counts:
        ax.hist(counts["after"], bins=bins, alpha=0.6, label="after training", color="#DD8452")
    ax.set_xlabel("# items assigned to a level-0 code")
    ax.set_ylabel("# codes (out of 256)")
    ax.set_title(f"Level-0 code-usage distribution — {dataset}\n"
                 f"(wider spread / heavier tail = less uniform usage)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIG_DIR, f"codebook_usage_hist_level0_{dataset}_before_after.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scientific")
    ap.add_argument("--config", default=None, help="dataset yaml (for histogram recompute)")
    ap.add_argument("--before_ckpt", default=None)
    ap.add_argument("--after_ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_hist", action="store_true", help="skip the recompute-based histogram")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    before = load_json(args.dataset, "before")
    after = load_json(args.dataset, "after")
    if before is None:
        raise SystemExit(f"[error] missing before JSON for {args.dataset}")
    after_avail = after is not None
    if not after_avail:
        print(f"[note] no AFTER json for {args.dataset} -> before-only figures "
              f"(after-training visualization pending until full run finishes)")
        after = before  # placeholder so per_level access works; after bars suppressed

    made = []
    made.append(fig_used(args.dataset, before, after, after_avail))
    made.append(fig_dead(args.dataset, before, after, after_avail))
    made.append(fig_entropy(args.dataset, before, after, after_avail))
    made.append(fig_top_codes(args.dataset, before, after, after_avail))

    if not args.no_hist:
        cfg = args.config or os.path.join(ROOT, "config", f"{args.dataset}.yaml")
        bck = args.before_ckpt
        if bck is None and os.path.exists(cfg):
            import yaml
            bck = yaml.safe_load(open(cfg)).get("rqvae_path")
        ack = args.after_ckpt
        if ack is None and after_avail:
            ack = after.get("meta", {}).get("checkpoint")
        try:
            counts = recompute_level_counts(args.dataset, cfg, bck, ack, device=args.device)
            h = fig_hist_level0(args.dataset, counts)
            if h:
                made.append(h)
        except Exception as e:  # keep JSON figures even if recompute fails
            print(f"[warn] histogram recompute skipped: {e}")

    print("[ok] figures written:")
    for m in made:
        if m:
            print("   ", os.path.relpath(m, ROOT))


if __name__ == "__main__":
    main()
