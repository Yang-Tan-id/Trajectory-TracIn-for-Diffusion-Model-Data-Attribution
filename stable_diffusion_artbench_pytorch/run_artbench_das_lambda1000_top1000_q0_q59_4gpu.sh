#!/usr/bin/env bash
set -euo pipefail

art_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$art_root"

python_bin="${PYTHON_BIN:-python}"
gpu_ids="${GPU_IDS:-0,1,2,3}"
query_ids="$(seq -s, 0 59)"
removal_root="${DAS_REMOVAL_ROOT:-$art_root/outputs/topk-removal-das-lambda1000-q0-59}"
metric_root="${DAS_METRIC_ROOT:-$art_root/outputs/topk-removal-das-lambda1000-q0-59/metrics}"

echo "[definition] ArtBench DAS lambda=1000 Top-1000 removal; Q0-Q59"
echo "[training] 100 epochs; remove 1000/5000; four-GPU static shards"
echo "[removal root] $removal_root"

"$python_bin" -m sd_artbench.launch_das_topk_removal \
  --gpu-ids "$gpu_ids" \
  --query-indices "$query_ids" \
  --top-k 1000 \
  --output-root "$removal_root"

echo "[evaluation] endpoint and 50-step DDIM trajectory"
CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID:-0}" "$python_bin" \
  -m sd_artbench.evaluate_topk_removal_trajectory_mse \
  --removal-root "$removal_root" \
  --output-dir "$metric_root" \
  --query-indices "$query_ids" \
  --methods das_lambda_1000 \
  --top-k 1000 \
  --steps 50

echo "[evaluation] Pixel-L2, CLIP, LPIPS and DINO"
CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID:-0}" "$python_bin" \
  -m sd_artbench.evaluate_topk_removal_perceptual \
  --image-root "$metric_root" \
  --query-indices "$query_ids" \
  --methods das_lambda_1000 \
  --batch-size "${PERCEPTUAL_BATCH_SIZE:-16}"

"$python_bin" -m sd_artbench.combine_das_removal_metrics \
  --input-dir "$metric_root" \
  --output "$metric_root/das_lambda1000_top1000_all_metrics_q0_q59.txt" \
  --expected-queries 60

echo "[done] $metric_root/das_lambda1000_top1000_all_metrics_q0_q59.txt"
