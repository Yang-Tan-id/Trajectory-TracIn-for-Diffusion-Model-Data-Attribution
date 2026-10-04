#!/usr/bin/env bash
#SBATCH -J 3d-paper-ret-tr
#SBATCH -o 3d-paper-ret-tr-%j.out
#SBATCH -e 3d-paper-ret-tr-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/replay_exact_training_interval.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
points="${ATTRIBUTION_POINTS:-5000}"
first="${FIRST_INTERVAL:-0}"
last="${LAST_INTERVAL:-48}"
result="$shapes/result/$experiment"
event_feature="${PAPER_EVENT_FEATURE:-raw_gradient_full_l2_normalized}"
artifact_name="${PAPER_ARTIFACT_NAME:-paper_retrac_full_l2_normalized_four_events_n${points}}"
artifact_root="$result/$artifact_name"
saved_root="$result/fixed_checkpoint_adamw_four_events_n${points}"
logs="$result/logs/paper_retrac_train_events/${SLURM_JOB_ID}"
mkdir -p "$artifact_root" "$logs"

echo "[definition] fixed-checkpoint saved training events; feature=$event_feature"
echo "[projection] checkpoint-specific seed shared with query; dim=4096"

for interval in $(seq "$first" "$last"); do
  start_epoch=$((4 * (interval + 1)))
  end_epoch=$((start_epoch + 4))
  out="$artifact_root/epoch_${start_epoch}_${end_epoch}"
  saved="$saved_root/epoch_${start_epoch}_${end_epoch}"
  log_dir="$logs/interval_${interval}"
  mkdir -p "$out" "$log_dir"

  complete=1
  for epoch in $(seq $((start_epoch + 1)) "$end_epoch"); do
    for shard in 0 1; do
      [[ -f "$out/event_gradient_epoch_$(printf '%04d' "$epoch")_shard_$(printf '%02d' "$shard")_of_02.npz" ]] || complete=0
    done
  done
  if (( complete )); then
    echo "[skip] interval=$interval already complete"
    continue
  fi

  pids=()
  for gpu in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="$gpu" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        python "$shapes/script/replay_exact_training_interval.py" \
          --experiment "$experiment" --train-seed "$seed" \
          --start-epoch "$start_epoch" --end-epoch "$end_epoch" \
          --saved-event-dir "$saved" --fixed-checkpoint \
          --extract-gradient-sketches \
          --event-feature "$event_feature" \
          --proj-dim 4096 --attribution-points "$points" \
          --shard-id "$gpu" --num-shards 2 --out-dir "$out"
    ) >"$log_dir/gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "interval $interval failed; inspect $log_dir" >&2; exit 1; }
  echo "[interval] $((interval + 1))/49 complete"
done

echo "[done] $artifact_root"
