#!/usr/bin/env bash
#SBATCH -J 3d-d20-score
#SBATCH -o 3d-d20-score-%j.out
#SBATCH -e 3d-d20-score-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/score_direction20_timestamp_square_checkpoint_major.py" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/diffusion_jax_refined/3dshapes/script/score_direction20_timestamp_square_checkpoint_major.py" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}

shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
train_artifact="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin_direction20_mean100t_adamw_dual/train_datapoint_gradient_artifact.npz"
score_namespace="recreate_adamw_full_direction20_mean100t_delta_l2normalized_timestamp_sum_squared_q0_99"
score_script="$shapes/script/score_direction20_timestamp_square_checkpoint_major.py"
lds_script="$shapes/script/run_traj_tracin_lds_cached.py"
log_root="$shapes/result/$experiment/logs/direction20_mean100t_scores/${SLURM_JOB_ID}"
mkdir -p "$log_root"

common=(
  --experiment "$experiment"
  --train-seed "$seed"
  --epochs "${JAX_EPOCHS:-200}"
  --query-file "$query_file"
  --query-ids "$query_ids"
  --train-artifact "$train_artifact"
  --score-namespace "$score_namespace"
  --direction-count 20
  --timestamp-count 100
  --query-batch-size "${DIRECTION_QUERY_BATCH_SIZE:-2}"
  --term-chunk-size "${DIRECTION_TERM_CHUNK_SIZE:-40}"
  --shard-count 2
)

echo "[score] streaming query gradients directly into checkpoint-major GPU contractions"
echo "[score] no query-gradient artifacts are written; each GPU owns one query shard"
echo "[score] AdamW full; normalized delta; sum over checkpoints for each (direction,t), then square and sum"
echo "[score] variants=raw,query_l2,train_l2,query_train_l2"
nvidia-smi

pids=()
for gpu in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
      "$python_bin" "$score_script" "${common[@]}" --shard-index "$gpu"
  ) >"$log_root/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "Score worker failed; inspect $log_root" >&2; exit 1; }

echo "[lds] cached four-target LDS for all four normalization variants"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  "$python_bin" "$lds_script" \
    --execute \
    --experiment "$experiment" \
    --train-seed "$seed" \
    --query-file "$query_file" \
    --query-ids "$query_ids" \
    --score-schemes "$score_namespace" \
    --prediction-sign 1 \
    >"$log_root/lds.log" 2>&1

echo "[done] direction20 mean100t AdamW-full scores and LDS"
