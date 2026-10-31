#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$HOME/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
art="$repo/stable_diffusion_artbench_pytorch"
py="${PYTHON_BIN:-$HOME/miniconda3/envs/sd_artbench/bin/python}"
gpu_csv="${GPU_IDS:-4,5,6,7}"
query_ids="${QUERY_IDS:-$(seq -s, 0 59)}"
lds_root="${LDS_ROOT:-$art/outputs/artbench2-lds}"
output_dir="${OUTPUT_DIR:-$lds_root/evaluation/four_targets_q0_59}"
full_lora_dir="${FULL_LORA_DIR:-$art/outputs/artbench2-lora/final}"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
log_dir="$output_dir/logs/$run_tag"

IFS=',' read -r -a gpus <<< "$gpu_csv"
(( ${#gpus[@]} > 0 )) || { echo "GPU_IDS is empty" >&2; exit 2; }
mkdir -p "$log_dir"
export PYTHONPATH="$art/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[definition] ArtBench true-f targets = endpoint, trajectory, shared-MC simple loss, reference-path predicted noise"
echo "[definition] simple-loss uses the same 50 deterministic probes for all 192 models per query"
echo "[definition] 192 models = 3 membership banks x 64 subsets; 2500/5000 kept"
echo "[queries] $query_ids"
echo "[gpus] $gpu_csv"
echo "[output] $output_dir"
echo "[logs] $log_dir"

pids=()
for slot in "${!gpus[@]}"; do
  gpu="${gpus[$slot]}"
  echo "[launch] slot=$slot/${#gpus[@]} gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$py" -m sd_artbench.artbench_lds_eval \
    --mode true-f \
    --lds-root "$lds_root" \
    --output-dir "$output_dir" \
    --full-lora-dir "$full_lora_dir" \
    --query-ids "$query_ids" \
    --model-shard-index "$slot" \
    --model-shard-count "${#gpus[@]}" \
    >"$log_dir/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for slot in "${!pids[@]}"; do
  if ! wait "${pids[$slot]}"; then
    echo "[failed] slot=$slot gpu=${gpus[$slot]} log=$log_dir/gpu_${gpus[$slot]}.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

"$py" -m sd_artbench.artbench_lds_eval \
  --mode merge \
  --lds-root "$lds_root" \
  --output-dir "$output_dir" \
  --query-ids "$query_ids" \
  | tee "$log_dir/merge.log"

"$py" -m sd_artbench.artbench_lds_eval \
  --mode lds \
  --lds-root "$lds_root" \
  --output-dir "$output_dir" \
  --query-ids "$query_ids" \
  | tee "$log_dir/lds.log"

echo "[complete] $output_dir"
