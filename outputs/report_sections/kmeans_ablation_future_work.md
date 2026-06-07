# K-means vs. Random Codebook Initialization — Future Work (Minh)

_This is the **dedicated initialization ablation**, which is **not done**. It is distinct from the
**completed** before-vs-after codebook visualization (`kmeans_codebook_visualization.md`)._

## What is done vs. not done
- **DONE — before/after codebook visualization.** We measured codebook utilization, dead codes,
  entropy, and collision rate of the **shipped random-init RQ-VAE** (`kmeans_init: False`) before
  vs. after end-to-end training on Scientific (plus Game before-only). Figures + captions:
  `kmeans_codebook_visualization.md`, `codebook_figure_captions.md`.
- **NOT DONE — random-init vs. k-means-init ablation.** We have **not** trained a second RQ-VAE
  with `kmeans_init: True` and compared it head-to-head against the random-init one. This requires
  re-pretraining the tokenizer and re-running the recommender, and has not been performed.

## Why it matters
K-means initialization seeds the codebook centroids on real clusters of the SASRec embeddings
(see `embedding_provenance.md`), which can improve **convergence stability** and **semantic
organization**, and in some settings reduces dead codes / collapse. Our before/after numbers show
the random-init codebook is **already ~100% utilized with high entropy** on these data, so we
**hypothesize** k-means init has little headroom to raise *utilization* here — but **this is a
hypothesis, not a measured result**, precisely because the ablation has not been run.

## Hypothesis to test
On Scientific (already-strong random-init baseline), k-means init is expected to:
1. give **comparable final utilization / dead-code counts** (random init is already near-ideal);
2. possibly **converge faster / more stably** during RQ-VAE pretraining;
3. produce **small, likely-within-noise** differences in downstream Test Recall@10 / NDCG@10.

## Proposed procedure
```bash
# 1) Re-pretrain the RQ-VAE with k-means init (Scientific):
cd RQVAE
# in run_pretrain.sh / the RQVAE config, set kmeans_init: True (and kmeans_iters: 100)
bash run_pretrain.sh                       # produces a new <...>.rqvae.pth

# 2) Repoint the recommender config at the k-means RQ-VAE:
#    config/scientific.yaml -> rqvae_path: ./dataset/scientific/<kmeans>.rqvae.pth
#    (or a copied config/scientific_kmeans.yaml to keep the random-init baseline intact)

# 3) Re-run the full schedule and parse:
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/scientific_kmeans.yaml --use_features=True --num_features=256 \
  --item_feature_path=item_features.npy 2>&1 | tee logs/etegrec_scientific_kmeans_$(date +%F_%H%M%S).log
python scripts/parse_training_results.py

# 4) Codebook utilization for the k-means tokenizer (before & after):
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_codebook_utilization.py \
  --config ./config/scientific_kmeans.yaml --tag before \
  --output outputs/codebook_utilization_scientific_kmeans_before.json
```

## Comparison to report
- **Headline metrics:** Test R@10 / N@10, k-means-init vs. random-init (same schedule, same seed).
- **Codebook health:** used/dead codes, normalized entropy, collision rate, both init schemes.
- **Convergence:** RQ-VAE pretrain loss curve and epochs-to-best.
- Run **≥2 seeds** if budget allows — the short-sweep results show differences at this scale are
  easily within single-seed noise (cf. the inconclusive Feature-Adapter result in
  `ablation_results.md`).

## Cost / blocker
Requires an extra RQ-VAE pretrain + a full ~8 h recommender run on the single GPU (CUDA 7).
Deferred for now; tracked here as future work. The current shipped codebooks use **random init**.
