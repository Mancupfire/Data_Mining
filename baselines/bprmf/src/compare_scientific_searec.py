#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = BASELINE_DIR.parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "scientific"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "bprmf_scientific_comparison.json"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass
class SplitData:
    user_index: dict
    item_index: dict
    train_histories: list
    valid_targets: np.ndarray
    test_histories: list
    test_targets: np.ndarray
    candidate_items: np.ndarray
    n_train_interactions: int
    n_test_interactions: int


def build_scientific_split(dataset_dir: Path) -> SplitData:
    valid_path = dataset_dir / "scientific.valid.jsonl"
    test_path = dataset_dir / "scientific.test.jsonl"
    map_path = dataset_dir / "scientific.emb_map.json"

    with map_path.open("r", encoding="utf-8") as handle:
        item_index = json.load(handle)

    valid_rows = list(load_jsonl(valid_path))
    test_rows = list(load_jsonl(test_path))
    if len(valid_rows) != len(test_rows):
        raise ValueError("Validation and test rows must align by user.")

    valid_by_user = {row["user_id"]: row for row in valid_rows}
    test_by_user = {row["user_id"]: row for row in test_rows}
    user_ids = sorted(valid_by_user.keys())
    if user_ids != sorted(test_by_user.keys()):
        raise ValueError("Validation/test user sets do not match.")

    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    train_histories = []
    valid_targets = np.zeros(len(user_ids), dtype=np.int32)
    test_histories = []
    test_targets = np.zeros(len(user_ids), dtype=np.int32)

    for user_id in user_ids:
        valid_row = valid_by_user[user_id]
        test_row = test_by_user[user_id]

        train_items = [item_index[token] for token in valid_row["inter_history"]]
        test_items = [item_index[token] for token in test_row["inter_history"]]
        train_histories.append(np.array(sorted(set(train_items)), dtype=np.int32))
        test_histories.append(np.array(sorted(set(test_items)), dtype=np.int32))
        valid_targets[user_index[user_id]] = item_index[valid_row["target_id"]]
        test_targets[user_index[user_id]] = item_index[test_row["target_id"]]

    candidate_items = np.arange(1, len(item_index), dtype=np.int32)
    n_train_interactions = int(sum(len(items) for items in train_histories))
    n_test_interactions = int(sum(len(items) for items in test_histories))

    return SplitData(
        user_index=user_index,
        item_index=item_index,
        train_histories=train_histories,
        valid_targets=valid_targets,
        test_histories=test_histories,
        test_targets=test_targets,
        candidate_items=candidate_items,
        n_train_interactions=n_train_interactions,
        n_test_interactions=n_test_interactions,
    )


def sample_negative(rng: np.random.Generator, positives_set: set[int], n_items: int) -> int:
    while True:
        candidate = int(rng.integers(1, n_items))
        if candidate not in positives_set:
            return candidate


def train_bpr_numpy(
    histories: list,
    eval_histories: list | None,
    eval_targets: np.ndarray | None,
    n_items: int,
    dim: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    reg: float,
    seed: int,
    early_stop: int,
    eval_interval: int,
    eval_batch_size: int,
):
    rng = np.random.default_rng(seed)
    n_users = len(histories)
    user_emb = rng.normal(0.0, 0.05, size=(n_users, dim)).astype(np.float32)
    item_emb = rng.normal(0.0, 0.05, size=(n_items, dim)).astype(np.float32)
    item_emb[0] = 0.0

    history_sets = [set(map(int, items.tolist())) for items in histories]
    non_empty_users = np.array([index for index, items in enumerate(histories) if len(items) > 0], dtype=np.int32)

    best = {"epoch": 0, "ndcg@10": -1.0, "user_emb": user_emb.copy(), "item_emb": item_emb.copy(), "valid_metrics": None}
    stale = 0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for _ in range(batches_per_epoch):
            users = rng.choice(non_empty_users, size=batch_size, replace=True)
            pos_items = np.empty(batch_size, dtype=np.int32)
            neg_items = np.empty(batch_size, dtype=np.int32)

            for idx, user in enumerate(users):
                positives = histories[user]
                pos_items[idx] = int(positives[rng.integers(0, len(positives))])
                neg_items[idx] = sample_negative(rng, history_sets[user], n_items)

            user_vec = user_emb[users].copy()
            pos_vec = item_emb[pos_items].copy()
            neg_vec = item_emb[neg_items].copy()

            x_uij = np.sum(user_vec * (pos_vec - neg_vec), axis=1)
            grad = 1.0 / (1.0 + np.exp(np.clip(x_uij, -50.0, 50.0)))
            grad_col = grad[:, None].astype(np.float32)

            user_update = grad_col * (pos_vec - neg_vec) - reg * user_vec
            pos_update = grad_col * user_vec - reg * pos_vec
            neg_update = -grad_col * user_vec - reg * neg_vec

            np.add.at(user_emb, users, learning_rate * user_update)
            np.add.at(item_emb, pos_items, learning_rate * pos_update)
            np.add.at(item_emb, neg_items, learning_rate * neg_update)

            total_loss += float(np.mean(np.log1p(np.exp(-x_uij))))

        if eval_targets is None:
            print(f"[epoch {epoch:03d}] loss={total_loss / batches_per_epoch:.4f}", flush=True)
            best = {
                "epoch": epoch,
                "ndcg@10": -1.0,
                "user_emb": user_emb.copy(),
                "item_emb": item_emb.copy(),
                "valid_metrics": None,
            }
            continue

        if epoch % eval_interval != 0:
            print(f"[epoch {epoch:03d}] loss={total_loss / batches_per_epoch:.4f}", flush=True)
            continue

        valid_metrics = evaluate_full_catalog(
            user_emb=user_emb,
            item_emb=item_emb,
            histories=eval_histories if eval_histories is not None else histories,
            targets=eval_targets,
            candidate_items=np.arange(1, n_items, dtype=np.int32),
            mask_history=False,
            batch_size=eval_batch_size,
        )
        print(
            f"[epoch {epoch:03d}] loss={total_loss / batches_per_epoch:.4f} "
            f"R@10={valid_metrics['recall@10']:.4f} N@10={valid_metrics['ndcg@10']:.4f}"
        , flush=True)

        if valid_metrics["ndcg@10"] > best["ndcg@10"]:
            best = {
                "epoch": epoch,
                "ndcg@10": valid_metrics["ndcg@10"],
                "user_emb": user_emb.copy(),
                "item_emb": item_emb.copy(),
                "valid_metrics": valid_metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= early_stop:
                break

    return best


def evaluate_full_catalog(user_emb, item_emb, histories, targets, candidate_items, mask_history, batch_size=1024):
    ks = [1, 5, 10]
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics.update({f"ndcg@{k}": 0.0 for k in ks})

    item_matrix = item_emb[candidate_items]
    item_id_to_position = {int(item_id): position for position, item_id in enumerate(candidate_items.tolist())}

    for start in range(0, len(histories), batch_size):
        end = min(start + batch_size, len(histories))
        batch_user = user_emb[start:end]
        scores = batch_user @ item_matrix.T

        for row, user_id in enumerate(range(start, end)):
            target_item = int(targets[user_id])
            if target_item not in item_id_to_position:
                continue
            if mask_history:
                positions = [item_id_to_position[item] for item in histories[user_id] if int(item) in item_id_to_position]
                if positions:
                    scores[row, positions] = -np.inf

            target_position = item_id_to_position[target_item]
            target_score = scores[row, target_position]
            rank = int(np.sum(scores[row] > target_score) + 1)

            for k in ks:
                if rank <= k:
                    metrics[f"recall@{k}"] += 1.0
                    metrics[f"ndcg@{k}"] += 1.0 / math.log2(rank + 1.0)

    total = float(len(histories))
    for key in list(metrics.keys()):
        metrics[key] = round(metrics[key] / total, 6)
    return metrics


def slice_subset(histories, targets, size, seed):
    if size is None or size <= 0 or size >= len(histories):
        return histories, targets
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(histories), size=size, replace=False))
    return [histories[index] for index in indices], targets[indices]


def load_searec_reference(root: Path):
    path = root / "outputs" / "results_scientific.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "recall@1": round(float(data["recall@1"]), 6),
        "recall@5": round(float(data["recall@5"]), 6),
        "recall@10": round(float(data["recall@10"]), 6),
        "ndcg@5": round(float(data["ndcg@5"]), 6),
        "ndcg@10": round(float(data["ndcg@10"]), 6),
        "best_epoch": int(data["best_epoch"]),
        "log": data["log"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run a BPR-MF comparison on SEARec Scientific.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--root-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--batches-per-epoch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--eval-sample-size", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    root_dir = args.root_dir.resolve()
    split = build_scientific_split(dataset_dir)
    valid_histories_eval, valid_targets_eval = slice_subset(
        split.train_histories,
        split.valid_targets,
        args.eval_sample_size,
        args.seed,
    )

    print(
        f"scientific users={len(split.user_index)} items={len(split.item_index) - 1} "
        f"train_interactions={split.n_train_interactions} test_context_interactions={split.n_test_interactions}"
    , flush=True)

    best = train_bpr_numpy(
        histories=split.train_histories,
        eval_histories=valid_histories_eval,
        eval_targets=valid_targets_eval,
        n_items=len(split.item_index),
        dim=args.dim,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reg=args.reg,
        seed=args.seed,
        early_stop=args.early_stop,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
    )

    refit = train_bpr_numpy(
        histories=split.test_histories,
        eval_histories=None,
        eval_targets=None,
        n_items=len(split.item_index),
        dim=args.dim,
        epochs=best["epoch"],
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reg=args.reg,
        seed=args.seed,
        early_stop=args.early_stop,
        eval_interval=1,
        eval_batch_size=args.eval_batch_size,
    )

    final_test_nomask = evaluate_full_catalog(
        user_emb=refit["user_emb"],
        item_emb=refit["item_emb"],
        histories=split.test_histories,
        targets=split.test_targets,
        candidate_items=split.candidate_items,
        mask_history=False,
        batch_size=args.eval_batch_size,
    )
    final_test_masked = evaluate_full_catalog(
        user_emb=refit["user_emb"],
        item_emb=refit["item_emb"],
        histories=split.test_histories,
        targets=split.test_targets,
        candidate_items=split.candidate_items,
        mask_history=True,
        batch_size=args.eval_batch_size,
    )

    searec_reference = load_searec_reference(root_dir)
    result = {
        "dataset": "scientific",
        "protocol": {
            "validation": "valid target, using valid.inter_history as observed training history",
            "test": "re-fit for selected epochs on test.inter_history, then rank the test target over the full catalog",
            "retrain_on_valid": True,
            "seed": args.seed,
        },
        "model": {
            "name": "bprmf_numpy",
            "dim": args.dim,
            "selected_epoch": best["epoch"],
            "refit_epochs": refit["epoch"],
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "batches_per_epoch": args.batches_per_epoch,
            "eval_interval": args.eval_interval,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "reg": args.reg,
        },
        "validation_best": best["valid_metrics"],
        "test_no_history_mask": final_test_nomask,
        "test_with_history_mask": final_test_masked,
        "searec_reference": searec_reference,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved comparison to {args.output}", flush=True)
    print("BPR-MF test (no history mask):", final_test_nomask, flush=True)
    if searec_reference is not None:
        print("SEARec reference:", searec_reference, flush=True)


if __name__ == "__main__":
    main()
