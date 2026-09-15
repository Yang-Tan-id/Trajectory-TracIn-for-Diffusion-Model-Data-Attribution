#!/usr/bin/env bash
#SBATCH -J 3d-loss-rms-p
#SBATCH -o 3d-loss-rms-p-%j.out
#SBATCH -e 3d-loss-rms-p-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

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
QUERY_DRIVER="${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_expected_residual_jacobian_scores.py"
TRAIN_LAUNCHER="${SHAPES_ROOT}/tacc/rtx_small/run_traj_tracin_loss_direction_residual_rms_train_rtx_small.sh"

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

TRAIN_NAMESPACE=traj_tracin_loss_direction_residual_rms
ORIGINAL_QUERY_NAMESPACE=loss_direction_residual_rms_original_f
PREDICTED_QUERY_PATTERN='loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}'
SCORE_PREFIX=traj_tracin_loss_direction_residual_rms
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/loss_direction_residual_rms_pipeline/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "3D Shapes loss-gradient direction x residual-RMS pipeline"
echo "train feature=RMS(residual) * unit(projected expected-loss gradient)"
echo "query targets=original f plus predicted-noise f with four probes"
echo "logs=${LOG_ROOT}"

echo "[phase 1/5] build/resume forward-only train-feature conversion"
bash "${TRAIN_LAUNCHER}"

echo "[phase 2/5] original-f query gradients"
"${PYTHON_BIN}" "${QUERY_DRIVER}" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 \
  --skip-sampling \
  --skip-score \
  --artifact-namespace "${ORIGINAL_QUERY_NAMESPACE}" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 10 \
  --python-bin "${PYTHON_BIN}"

echo "[phase 3/5] four predicted-noise query-probe banks"
for probe_index in 0 1 2 3; do
  namespace="loss_direction_residual_rms_predicted_noise_probe4_r${probe_index}"
  "${PYTHON_BIN}" "${QUERY_DRIVER}" \
    --execute \
    --experiment "${EXPERIMENT_TAG}" \
    --train-seed "${TRAIN_SEED}" \
    --epochs "${JAX_EPOCHS}" \
    --query-ids 0,1,2,3,4,5,6,7,8,9 \
    --gpus 0,1 \
    --skip-sampling \
    --skip-score \
    --artifact-namespace "${namespace}" \
    --query-objective trajectory_predicted_noise_probe \
    --predicted-noise-probe-index "${probe_index}" \
    --num-snapshots 10 \
    --python-bin "${PYTHON_BIN}"
done

echo "[phase 4/5] two checkpoint score shards and merge"
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
        --predicted-query-namespace "${PREDICTED_QUERY_PATTERN}" \
        --predicted-num-probes 4 \
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
  --predicted-query-namespace "${PREDICTED_QUERY_PATTERN}" \
  --predicted-num-probes 4 \
  --score-namespace-prefix "${SCORE_PREFIX}"

echo "[phase 5/5] cached LDS: two f targets x raw/query-L2"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --score-schemes loss_direction_residual_rms_original_f,loss_direction_residual_rms_predicted_noise

echo "[done] original f plus four-probe predicted-noise f scores and LDS are complete"
