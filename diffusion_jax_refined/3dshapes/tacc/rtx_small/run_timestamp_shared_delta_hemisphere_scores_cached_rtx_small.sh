#!/usr/bin/env bash
#SBATCH -J 3d-dhemi
#SBATCH -o 3d-dhemi-%j.out
#SBATCH -e 3d-dhemi-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_predicted_noise_angle_oriented_scores.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SEEDS=(
  314159265 271828183 161803399 141421357
  538102947 794615203 126937481 682450719
  905173624 417286953 263590817 849031576
)
SEEDS_CSV="$(IFS=,; echo "${SEEDS[*]}")"
patterns=()
for seed in "${SEEDS[@]}"; do
  patterns+=("loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${seed}_r0")
done
PATTERNS_CSV="$(IFS=,; echo "${patterns[*]}")"

echo "[score] original angle-sign/weighted plus post-normalization delta-norm-weighted versions"
CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
python "${SHAPES_ROOT}/script/analyze_predicted_noise_angle_oriented_scores.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --num-probes 12 --run-id "${SLURM_JOB_ID}" --prediction-sign 1 \
  --query-namespace-patterns "${PATTERNS_CSV}" \
  --expected-query-probe-mode timestamp_shared_gaussian \
  --expected-query-probe-seeds "${SEEDS_CSV}" \
  --alignment-source timestamp_shared_delta \
  --geometry-namespace reference_probe_delta_geometry_collect_all

echo "[done] delta-hemisphere-oriented timestamp-shared probe scores"
