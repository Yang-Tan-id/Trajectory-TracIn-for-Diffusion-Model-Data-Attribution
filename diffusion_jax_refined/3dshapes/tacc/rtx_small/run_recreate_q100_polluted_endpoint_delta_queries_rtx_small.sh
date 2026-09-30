#!/usr/bin/env bash
#SBATCH -J 3d-q100-pdelta
#SBATCH -o 3d-q100-pdelta-%j.out
#SBATCH -e 3d-q100-pdelta-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -A IRI26004

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-10}"
export TRAJ_QUERY_USE_CONFIG_SNAPSHOTS=1

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_ids="${QUERY_IDS:-100}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
raw_namespace="recreate_q100_polluted_endpoint_next_delta_raw_10t"
normalized_namespace="recreate_q100_polluted_endpoint_next_delta_l2normalized_10t"
log_root="$shapes/result/$experiment/logs/recreate_q100_polluted_endpoint_delta/${SLURM_JOB_ID}"
mkdir -p "$log_root"

run_query() {
  local gpu="$1" namespace="$2" objective="$3"
  CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    python "$shapes/script/run_traj_tracin_queries_and_scores.py" \
      --execute --experiment "$experiment" --train-seed "$seed" \
      --epochs "${JAX_EPOCHS:-200}" \
      --query-file "$query_file" --query-ids "$query_ids" --gpus "$gpu" \
      --skip-sampling --skip-score \
      --artifact-namespace "$namespace" \
      --query-objective "$objective" \
      --num-snapshots 10 \
      --log-prefix "${namespace}"
}

echo "[query] Q${query_ids}; saved endpoint uses train-aligned noise per checkpoint/timestamp"
echo "[query] x_t differs across checkpoints; same key is used by train/raw/normalized delta"
echo "[sample] ensure Q${query_ids} endpoint/trajectory exists before parallel query workers"
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
  python "$shapes/script/run_traj_tracin_queries_and_scores.py" \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --epochs "${JAX_EPOCHS:-200}" \
    --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
    --skip-query-gradient --skip-score
run_query 0 "$raw_namespace" \
  trajectory_polluted_endpoint_next_delta_projection \
  >"$log_root/raw_delta.log" 2>&1 &
pid0="$!"
run_query 1 "$normalized_namespace" \
  trajectory_polluted_endpoint_next_delta_projection_normalized \
  >"$log_root/normalized_delta.log" 2>&1 &
pid1="$!"

failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
(( failed == 0 )) || { echo "query worker failed; inspect $log_root" >&2; exit 1; }

echo "[done] raw delta namespace=$raw_namespace"
echo "[done] L2-normalized delta namespace=$normalized_namespace"
