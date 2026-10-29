#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$HOME/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
celeba="$repo/diffusion_jax_refined/celeba"
py="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
query_file="${QUERY_FILE:-$celeba/queries_attr40_val25_x4.json}"
root="${OUTPUT_ROOT:-$celeba/result/attr40_experiment1/analysis/random_cross_model_noise_correlation_q0_9}"
gpu_text="${GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a gpus <<< "$gpu_text"
[[ "${#gpus[@]}" -eq 4 ]] || { echo "GPU_IDS must contain four GPUs"; exit 2; }

mkdir -p "$root/logs"
query_groups=("0,1,2" "3,4,5" "6,7" "8,9")
model_ids="$(seq -s, 0 69)"
pids=()

for slot in 0 1 2 3; do
  echo "[launch] slot=$slot gpu=${gpus[$slot]} queries=${query_groups[$slot]}"
  CUDA_VISIBLE_DEVICES="${gpus[$slot]}" "$py" \
    "$celeba/script/analyze_random_removal_noise_correlation.py" \
    --query-file "$query_file" \
    --query-ids "${query_groups[$slot]}" \
    --model-ids "$model_ids" \
    --output-dir "$root/shard_$slot" \
    > "$root/logs/gpu_${gpus[$slot]}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for slot in 0 1 2 3; do
  if wait "${pids[$slot]}"; then
    echo "[complete] slot=$slot gpu=${gpus[$slot]}"
  else
    echo "[failed] slot=$slot gpu=${gpus[$slot]}"
    failed=1
  fi
done
exit "$failed"
