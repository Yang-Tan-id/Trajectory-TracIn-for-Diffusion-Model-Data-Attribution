#!/usr/bin/env bash
#SBATCH -J 3d-ret-tsq
#SBATCH -o 3d-ret-tsq-%j.out
#SBATCH -e 3d-ret-tsq-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/score_retrac_endpoint100x1_checkpoint_major.py" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/diffusion_jax_refined/3dshapes/script/score_retrac_endpoint100x1_checkpoint_major.py" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}

shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
python_bin="${PYTHON_BIN:-python}"
experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
retrac_root="$shapes/result/$experiment/fixed_checkpoint_raw_four_events_n5000"
namespace="retrac_exact4_endpoint100x1_timestamp_sum_squared_q0_99"
logs="$shapes/result/$experiment/logs/retrac_endpoint100x1_timestamp_sum_squared/${SLURM_JOB_ID}"
mkdir -p "$logs"

event_count="$(find "$retrac_root" -type f -name 'event_gradient_epoch_*_shard_*_of_02.npz' | wc -l | tr -d ' ')"
[[ "$event_count" == 392 ]] || {
  echo "Expected 392 exact raw ReTrac event parts (49x4x2), found $event_count" >&2
  exit 1
}

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "[definition] group exact ReTrac event terms by replayed training timestep"
echo "[reduction] sum checkpoints/events within timestep, square, then sum timestamps"
echo "[query] endpoint denoising mean loss, 100 timestamps x MC1"

pids=()
for gpu in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
      "$python_bin" "$shapes/script/score_retrac_endpoint100x1_checkpoint_major.py" \
        --query-file "$query_file" \
        --query-ids "$query_ids" \
        --experiment "$experiment" \
        --train-seed "$seed" \
        --retrac-event-root "$retrac_root" \
        --methods retrac \
        --retrac-reduction timestamp_sum_squared \
        --retrac-namespace "$namespace" \
        --query-batch-size "${QUERY_BATCH_SIZE:-2}" \
        --shard-index "$gpu" --shard-count 2
  ) >"$logs/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "score worker failed; inspect $logs" >&2; exit 1; }

echo "[LDS] cached true-f, loss utility sign=-1"
JAX_PLATFORMS=cpu "$python_bin" "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "$experiment" \
  --train-seed "$seed" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --score-schemes "$namespace" \
  --prediction-sign -1

echo "[done] Q0-Q99 ReTrac exact-event timestamp-wise-square plus LDS"
