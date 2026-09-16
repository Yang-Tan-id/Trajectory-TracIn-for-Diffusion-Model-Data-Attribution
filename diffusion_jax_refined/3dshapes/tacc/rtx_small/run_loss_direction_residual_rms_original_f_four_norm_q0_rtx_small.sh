#!/usr/bin/env bash
#SBATCH -J 3d-orgf4n-q0
#SBATCH -o 3d-orgf4n-q0-%j.out
#SBATCH -e 3d-orgf4n-q0-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
resolve_repo_root() {
  local start candidate
  for start in "${REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}"; do
    [[ -n "${start}" && -d "${start}" ]] || continue
    candidate="$(cd "${start}" && pwd)"
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_expected_residual_jacobian_scores.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || { echo "Could not locate repository" >&2; exit 1; }
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_expected_residual_jacobian_scores.py"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

TRAIN_NAMESPACE=traj_tracin_loss_direction_residual_rms
ORIGINAL_QUERY_NAMESPACE=loss_direction_residual_rms_original_f
SCORE_PREFIX=traj_tracin_loss_direction_residual_rms
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/original_f_four_norm_q0/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "[phase 1/2] original-f four-normalization scores from cached train/query artifacts"
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
        --experiment "${EXPERIMENT_TAG}" \
        --train-seed "${TRAIN_SEED}" \
        --epochs "${JAX_EPOCHS}" \
        --run-id "${SLURM_JOB_ID}" \
        --shard-index "${shard}" \
        --shard-count 2 \
        --train-namespace "${TRAIN_NAMESPACE}" \
        --train-feature-semantics unit_projected_expected_loss_gradient_times_matching_mc_residual_rms \
        --train-feature-description residual_RMS_times_unit_projected_expected_loss_gradient \
        --original-query-namespace "${ORIGINAL_QUERY_NAMESPACE}" \
        --original-contraction signed \
        --skip-predicted \
        --score-namespace-prefix "${SCORE_PREFIX}"
  ) >"${LOG_ROOT}/score_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || { echo "Score shard failed; inspect ${LOG_ROOT}" >&2; exit 1; }

"${PYTHON_BIN}" "${SCORE_DRIVER}" merge \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --run-id "${SLURM_JOB_ID}" \
  --shard-count 2 \
  --train-namespace "${TRAIN_NAMESPACE}" \
  --train-feature-semantics unit_projected_expected_loss_gradient_times_matching_mc_residual_rms \
  --train-feature-description residual_RMS_times_unit_projected_expected_loss_gradient \
  --original-query-namespace "${ORIGINAL_QUERY_NAMESPACE}" \
  --original-contraction signed \
  --skip-predicted \
  --score-namespace-prefix "${SCORE_PREFIX}"

echo "[phase 2/2] Q0 cached LDS for raw/query-L2/train-L2/both-L2"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids 0 \
  --score-schemes loss_direction_residual_rms_original_f \
  --prediction-sign=-1

echo "[done] Q0 original-f four-normalization LDS complete"
