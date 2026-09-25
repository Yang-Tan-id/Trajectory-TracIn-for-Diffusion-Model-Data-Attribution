#!/usr/bin/env bash
#SBATCH -J 3d-q0-29-qtsv
#SBATCH -o 3d-q0-29-qtsv-%j.out
#SBATCH -e 3d-q0-29-qtsv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

query_namespace="loss_direction_predicted_noise_probe1_query_timestamp_shared_seed20260925_reference_10t"

cd "$shapes"
echo '[query] Q0-Q29: one query-specific v(q,t), shared across every checkpoint'
python script/run_traj_tracin_queries_and_scores.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --epochs "${JAX_EPOCHS:-200}" \
  --query-file queries_in_distribution_plus_zero_seed_100_219.json \
  --query-ids "$(seq -s, 0 29)" \
  --gpus 0,1 \
  --skip-sampling \
  --skip-score \
  --artifact-namespace "$query_namespace" \
  --query-objective trajectory_predicted_noise_probe \
  --predicted-noise-probe-index 0 \
  --predicted-noise-probe-count 1 \
  --predicted-noise-probe-mode query_timestamp_shared_gaussian \
  --predicted-noise-probe-seed 20260925 \
  --num-snapshots 10 \
  --log-prefix q0_29_query_timestamp_shared

echo "[done] Q0-Q29 query artifacts: $query_namespace"
