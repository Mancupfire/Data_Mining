#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from compare_scientific_searec import (
    build_scientific_split,
    evaluate_full_catalog,
    load_searec_reference,
    train_bpr_numpy,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = BASELINE_DIR.parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "scientific"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs" / "bprmf_scientific_suite.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "outputs" / "bprmf_scientific_suite.md"


def run_one(split, config, seed):
    best = train_bpr_numpy(
        histories=split.train_histories,
        eval_histories=split.train_histories,
        eval_targets=split.valid_targets,
        n_items=len(split.item_index),
        dim=config["dim"],
        epochs=config["epochs"],
        batches_per_epoch=config["batches_per_epoch"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        reg=config["reg"],
        seed=seed,
        early_stop=config["early_stop"],
        eval_interval=config["eval_interval"],
        eval_batch_size=config["eval_batch_size"],
    )

    refit = train_bpr_numpy(
        histories=split.test_histories,
        eval_histories=None,
        eval_targets=None,
        n_items=len(split.item_index),
        dim=config["dim"],
        epochs=best["epoch"],
        batches_per_epoch=config["batches_per_epoch"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        reg=config["reg"],
        seed=seed,
        early_stop=config["early_stop"],
        eval_interval=1,
        eval_batch_size=config["eval_batch_size"],
    )

    test_metrics = evaluate_full_catalog(
        user_emb=refit["user_emb"],
        item_emb=refit["item_emb"],
        histories=split.test_histories,
        targets=split.test_targets,
        candidate_items=split.candidate_items,
        mask_history=False,
        batch_size=config["eval_batch_size"],
    )
    return {
        "seed": seed,
        "config": config,
        "selected_epoch": best["epoch"],
        "validation_best": best["valid_metrics"],
        "test": test_metrics,
    }


def mean_std(rows, key):
    values = [row["test"][key] for row in rows]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return round(mean, 6), round(variance ** 0.5, 6)


def write_markdown(path: Path, summary, searec_reference):
    lines = [
        "# Scientific BPR-MF vs SEARec",
        "",
        "## Best BPR-MF configuration",
        "",
        f"- `dim={summary['best_config']['dim']}`, `lr={summary['best_config']['learning_rate']}`, `reg={summary['best_config']['reg']}`",
        f"- `epochs={summary['best_config']['epochs']}`, `batch_size={summary['best_config']['batch_size']}`, `batches_per_epoch={summary['best_config']['batches_per_epoch']}`",
        "",
        "## Multi-seed test comparison",
        "",
        "| Model | R@1 | R@5 | R@10 | N@5 | N@10 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| BPR-MF (mean ± std, {len(summary['final_runs'])} seeds) | "
            f"{summary['aggregate']['recall@1'][0]:.4f} ± {summary['aggregate']['recall@1'][1]:.4f} | "
            f"{summary['aggregate']['recall@5'][0]:.4f} ± {summary['aggregate']['recall@5'][1]:.4f} | "
            f"{summary['aggregate']['recall@10'][0]:.4f} ± {summary['aggregate']['recall@10'][1]:.4f} | "
            f"{summary['aggregate']['ndcg@5'][0]:.4f} ± {summary['aggregate']['ndcg@5'][1]:.4f} | "
            f"{summary['aggregate']['ndcg@10'][0]:.4f} ± {summary['aggregate']['ndcg@10'][1]:.4f} |"
        ),
    ]
    if searec_reference is not None:
        lines.append(
            f"| SEARec (saved reference) | {searec_reference['recall@1']:.4f} | {searec_reference['recall@5']:.4f} | {searec_reference['recall@10']:.4f} | {searec_reference['ndcg@5']:.4f} | {searec_reference['ndcg@10']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed BPR-MF runs",
            "",
            "| Seed | Best epoch | R@10 | N@10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["final_runs"]:
        lines.append(
            f"| {row['seed']} | {row['selected_epoch']} | {row['test']['recall@10']:.4f} | {row['test']['ndcg@10']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Tune and run multi-seed BPR-MF on Scientific.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--root-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    split = build_scientific_split(args.dataset_dir.resolve())
    searec_reference = load_searec_reference(args.root_dir.resolve())

    grid = [
        {"dim": 64, "learning_rate": 0.05, "reg": 1e-4, "epochs": 12, "batch_size": 2048, "batches_per_epoch": 64, "early_stop": 3, "eval_interval": 2, "eval_batch_size": 1024},
        {"dim": 128, "learning_rate": 0.05, "reg": 1e-4, "epochs": 12, "batch_size": 2048, "batches_per_epoch": 64, "early_stop": 3, "eval_interval": 2, "eval_batch_size": 1024},
        {"dim": 64, "learning_rate": 0.02, "reg": 1e-4, "epochs": 12, "batch_size": 2048, "batches_per_epoch": 64, "early_stop": 3, "eval_interval": 2, "eval_batch_size": 1024},
        {"dim": 128, "learning_rate": 0.02, "reg": 1e-5, "epochs": 12, "batch_size": 2048, "batches_per_epoch": 64, "early_stop": 3, "eval_interval": 2, "eval_batch_size": 1024},
    ]

    tuning_seed = 2020
    tuning_runs = []
    best_idx = None
    best_score = -1.0
    for index, config in enumerate(grid, start=1):
        print(f"\n=== tuning config {index}/{len(grid)}: {config} ===", flush=True)
        result = run_one(split, config, tuning_seed)
        tuning_runs.append(result)
        score = result["validation_best"]["ndcg@10"]
        if score > best_score:
            best_score = score
            best_idx = index - 1

    best_config = grid[best_idx]
    print(f"\n=== best config: {best_config} ===", flush=True)

    final_seeds = [2020, 2021, 2022]
    final_runs = []
    final_config = dict(best_config)
    final_config["epochs"] = 16
    final_config["early_stop"] = 4
    final_config["eval_interval"] = 2
    for seed in final_seeds:
        print(f"\n=== final seed {seed} ===", flush=True)
        final_runs.append(run_one(split, final_config, seed))

    aggregate = {}
    for key in ["recall@1", "recall@5", "recall@10", "ndcg@5", "ndcg@10"]:
        aggregate[key] = mean_std(final_runs, key)

    summary = {
        "dataset": "scientific",
        "tuning_seed": tuning_seed,
        "tuning_runs": tuning_runs,
        "best_config": best_config,
        "final_config": final_config,
        "final_runs": final_runs,
        "aggregate": aggregate,
        "searec_reference": searec_reference,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary, searec_reference)
    print(f"saved suite json to {args.output_json}", flush=True)
    print(f"saved suite markdown to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
