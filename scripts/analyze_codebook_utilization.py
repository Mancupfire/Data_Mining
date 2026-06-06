"""Measure RQ-VAE codebook utilization for ETEGRec.

The item tokenizer in ETEGRec is the residual VQ-VAE defined in ``vq.RQVAE``.
This script loads an RQ-VAE checkpoint, assigns every item's semantic
(SASRec) embedding to its codebook entries, and reports per-level utilization
statistics. Run it once on the *pretrained* RQ-VAE (``before`` training) and
once on the *end-to-end trained* tokenizer (``after`` training) to compare.

Examples
--------
# "before": pretrained RQ-VAE shipped with the dataset
python scripts/analyze_codebook_utilization.py \
    --config ./config/scientific.yaml --tag before

# "after": tokenizer saved during end-to-end training
python scripts/analyze_codebook_utilization.py \
    --config ./config/scientific.yaml --tag after \
    --rqvae_ckpt ./myckpt/scientific/<run>/<epoch>.pt.rqvae

# validate the metric code path without any real data
python scripts/analyze_codebook_utilization.py --self_test

Both real runs append to ``outputs/codebook_utilization_{dataset}.json`` so the
``before`` and ``after`` results live side by side.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import yaml

# allow running from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vq import RQVAE  # noqa: E402


def codebook_metrics(indices, total_codes):
    """Compute utilization stats for one codebook level.

    Args:
        indices: 1-D numpy array of assigned code ids for this level.
        total_codes: codebook size (n_e) for this level.
    """
    counts = np.bincount(indices, minlength=total_codes).astype(np.int64)
    used_mask = counts > 0
    used_codes = int(used_mask.sum())
    dead_codes = int(total_codes - used_codes)

    probs = counts / counts.sum()
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum())            # nats
    max_entropy = float(np.log(total_codes))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    order = np.argsort(-counts)                            # most used first
    top10 = [[int(c), int(counts[c])] for c in order[:10]]
    used_order = order[counts[order] > 0]
    bottom10 = [[int(c), int(counts[c])] for c in used_order[::-1][:10]]

    return {
        "total_codes": int(total_codes),
        "used_codes": used_codes,
        "utilization_rate": round(used_codes / total_codes, 6),
        "dead_codes": dead_codes,
        "assignment_entropy": round(entropy, 6),
        "normalized_entropy": round(norm_entropy, 6),
        "top_10_most_used_codes": top10,
        "bottom_10_least_used_codes": bottom10,
    }


def analyze(all_indices, n_e_list):
    """all_indices: (N, L) int array of per-level code ids."""
    all_indices = np.asarray(all_indices)
    n_items, n_levels = all_indices.shape
    per_level = {}
    for lvl in range(n_levels):
        total = n_e_list[lvl] if lvl < len(n_e_list) else int(all_indices[:, lvl].max() + 1)
        per_level[f"level_{lvl}"] = codebook_metrics(all_indices[:, lvl], total)

    # joint code (tuple) collision stats
    tuples = [" ".join(map(str, row)) for row in all_indices.tolist()]
    unique_tuples = len(set(tuples))
    joint = {
        "num_items": int(n_items),
        "unique_semantic_ids": int(unique_tuples),
        "collision_rate": round(1.0 - unique_tuples / n_items, 6),
    }
    return {"per_level": per_level, "joint": joint}


def build_config(config_path):
    config = yaml.safe_load(open(config_path, "r"))
    return config


def load_rqvae(config, ckpt_path, device):
    in_dim = config["semantic_hidden_size"]
    model = RQVAE(config=config, in_dim=in_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # the model_id (.pt.rqvae) and standalone rqvae .pth share the RQVAE keys
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys when loading {ckpt_path}: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading {ckpt_path}: {unexpected}")
    model.to(device).eval()
    return model


def load_embeddings(config):
    data_path = config["data_path"]
    dataset = config["dataset"]
    emb_path = os.path.join(data_path, dataset, config["semantic_emb_path"])
    try:
        emb = np.load(emb_path)
    except ValueError as e:
        raise SystemExit(
            f"[error] could not read '{emb_path}': {e}\n"
            f"        The semantic embedding file looks truncated/corrupted. "
            f"Re-download the full *_emb_256.npy before running utilization analysis."
        )
    return emb.astype(np.float32)


@torch.no_grad()
def get_all_indices(model, emb, device, batch_size=4096):
    idx_chunks = []
    for i in range(0, len(emb), batch_size):
        x = torch.from_numpy(emb[i:i + batch_size]).to(device)
        idx = model.get_indices(x).cpu().numpy()
        idx_chunks.append(idx)
    return np.concatenate(idx_chunks, axis=0)


def write_result(out_path, tag, payload):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {}
    if os.path.exists(out_path):
        try:
            data = json.load(open(out_path))
        except Exception:
            data = {}
    data[tag] = payload
    json.dump(data, open(out_path, "w"), indent=2)
    print(f"[ok] wrote {tag} results -> {out_path}")


def run_self_test():
    """Exercise the metric path with synthetic data (no real files needed)."""
    print("[self_test] building random RQ-VAE + random embeddings ...")
    config = {
        "semantic_hidden_size": 256, "e_dim": 128, "layers": [512, 256],
        "dropout_prob": 0.0, "bn": False, "alpha": 1, "beta": 0.25,
        "vq_type": "vq", "num_emb_list": [256, 256, 256], "dist": "l2",
        "kmeans_init": False, "kmeans_iters": 100,
    }
    device = "cpu"
    model = RQVAE(config=config, in_dim=256).to(device).eval()
    emb = np.random.randn(2000, 256).astype(np.float32)
    indices = get_all_indices(model, emb, device)
    result = analyze(indices, config["num_emb_list"])
    print(json.dumps(result, indent=2))
    assert set(result["per_level"]["level_0"]) >= {
        "total_codes", "used_codes", "utilization_rate", "dead_codes",
        "assignment_entropy", "top_10_most_used_codes", "bottom_10_least_used_codes",
    }
    print("[self_test] PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="dataset config yaml")
    parser.add_argument("--rqvae_ckpt", type=str, default=None,
                        help="RQ-VAE checkpoint; default = config['rqvae_path'] (pretrained)")
    parser.add_argument("--tag", type=str, default="before",
                        help="label for this run, e.g. 'before' or 'after'")
    parser.add_argument("--output", type=str, default=None,
                        help="output json; default outputs/codebook_utilization_{dataset}.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.config:
        parser.error("--config is required (or use --self_test)")

    config = build_config(args.config)
    dataset = config["dataset"]
    ckpt = args.rqvae_ckpt or config.get("rqvae_path")
    if ckpt is None or not os.path.exists(ckpt):
        raise SystemExit(f"[error] RQ-VAE checkpoint not found: {ckpt}")

    out_path = args.output or f"./outputs/codebook_utilization_{dataset}.json"

    print(f"[info] dataset={dataset} tag={args.tag} ckpt={ckpt} device={args.device}")
    model = load_rqvae(config, ckpt, args.device)
    emb = load_embeddings(config)
    print(f"[info] {len(emb)} item embeddings, dim={emb.shape[1]}")
    indices = get_all_indices(model, emb, args.device)
    result = analyze(indices, config["num_emb_list"])
    result["meta"] = {"dataset": dataset, "checkpoint": ckpt, "n_items": int(len(emb))}

    print(json.dumps(result["per_level"], indent=2))
    print(json.dumps(result["joint"], indent=2))
    write_result(out_path, args.tag, result)


if __name__ == "__main__":
    main()
