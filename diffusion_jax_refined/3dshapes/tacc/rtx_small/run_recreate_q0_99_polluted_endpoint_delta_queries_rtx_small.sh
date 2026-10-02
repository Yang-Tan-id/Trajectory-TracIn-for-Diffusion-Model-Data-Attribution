#!/usr/bin/env bash
#SBATCH -J 3d-q0-99-pdelta
#SBATCH -o 3d-q0-99-pdelta-%j.out
#SBATCH -e 3d-q0-99-pdelta-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }

export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-10}"
export TRAJ_QUERY_USE_CONFIG_SNAPSHOTS=1

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
raw_namespace="recreate_q0_99_polluted_endpoint_next_delta_raw_10t"
normalized_namespace="recreate_q0_99_polluted_endpoint_next_delta_l2normalized_10t"
log_root="$shapes/result/$experiment/logs/recreate_q0_99_polluted_endpoint_delta/${SLURM_JOB_ID}"
mkdir -p "$log_root"

echo "[sample] verify/reuse Q0-Q99 saved endpoint trajectories on both GPUs"
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
  "$python_bin" "$shapes/script/run_traj_tracin_queries_and_scores.py" \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --epochs "${JAX_EPOCHS:-200}" \
    --query-file "$query_file" --query-ids "$query_ids" --gpus 0,1 \
    --python-bin "$python_bin" \
    --skip-query-gradient --skip-score

bank_script="$shapes/script/generate_polluted_delta_query_bank_checkpoint_major.py"
common_args=(
  --experiment "$experiment"
  --train-seed "$seed"
  --epochs "${JAX_EPOCHS:-200}"
  --query-file "$query_file"
  --query-ids "$query_ids"
  --raw-namespace "$raw_namespace"
  --normalized-namespace "$normalized_namespace"
  --batch-size "${QUERY_BATCH_SIZE:-2}"
  --shard-count 2
)

echo "[query] checkpoint-major Q0-Q99; one checkpoint pair restore serves all queries"
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
  "$python_bin" "$bank_script" "${common_args[@]}" --shard-index 0 \
  >"$log_root/checkpoint_shard_0.log" 2>&1 &
pid0="$!"
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
  "$python_bin" "$bank_script" "${common_args[@]}" --shard-index 1 \
  >"$log_root/checkpoint_shard_1.log" 2>&1 &
pid1="$!"

failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
(( failed == 0 )) || { echo "query worker failed; inspect $log_root" >&2; exit 1; }

JAX_PLATFORMS=cpu "$python_bin" "$bank_script" \
  "${common_args[@]}" --merge-only \
  >"$log_root/merge.log" 2>&1

echo "[done] Q0-Q99 raw delta namespace=$raw_namespace"
echo "[done] Q0-Q99 L2-normalized delta namespace=$normalized_namespace"
