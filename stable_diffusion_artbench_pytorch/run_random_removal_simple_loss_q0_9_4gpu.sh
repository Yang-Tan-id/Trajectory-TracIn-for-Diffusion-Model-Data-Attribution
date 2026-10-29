#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$HOME/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
art="$repo/stable_diffusion_artbench_pytorch"
py="${PYTHON_BIN:-$HOME/miniconda3/envs/sd_artbench/bin/python}"
removal_root="${REMOVAL_ROOT:-$art/outputs/topk-removal-five-method-metrics-q0-59/combined}"
noise_root="${NOISE_ROOT:-$art/outputs/random-removal-noise-correlation-cross-q0-59}"
output_root="${OUTPUT_ROOT:-$art/outputs/random-removal-simple-loss-correlation-q0-9}"
gpu_text="${GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a gpus <<< "$gpu_text"
[[ "${#gpus[@]}" -eq 4 ]] || { echo "GPU_IDS must contain four GPUs"; exit 2; }

export PYTHONPATH="$art/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$output_root/logs"
query_groups=("0,1,2" "3,4,5" "6,7" "8,9")
model_indices="$(seq -s, 0 59)"
pids=()

for slot in 0 1 2 3; do
  shard="$output_root/shard_$slot"
  mkdir -p "$shard"
  echo "[launch] gpu=${gpus[$slot]} queries=${query_groups[$slot]}"
  CUDA_VISIBLE_DEVICES="${gpus[$slot]}" "$py" -m \
    sd_artbench.analyze_random_removal_simple_loss_correlation \
    --removal-root "$removal_root" \
    --noise-correlation-root "$noise_root" \
    --query-indices "${query_groups[$slot]}" \
    --model-indices "$model_indices" \
    --output-dir "$shard" \
    > "$output_root/logs/gpu_${gpus[$slot]}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for slot in 0 1 2 3; do
  if wait "${pids[$slot]}"; then
    echo "[complete] gpu=${gpus[$slot]}"
  else
    echo "[failed] gpu=${gpus[$slot]}"
    failed=1
  fi
done

if [[ "$failed" -eq 0 ]]; then
  "$py" -m sd_artbench.merge_random_removal_simple_loss \
    --root "$output_root" \
    --query-indices "$(seq -s, 0 9)" \
    --model-indices "$model_indices"
fi
exit "$failed"
