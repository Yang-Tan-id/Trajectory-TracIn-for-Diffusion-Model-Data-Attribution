#!/usr/bin/env bash
#SBATCH -J 3d-pn8-sqplay
#SBATCH -o 3d-pn8-sqplay-%j.out
#SBATCH -e 3d-pn8-sqplay-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"
[[ -f "${SCORE_DRIVER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

NUM_PROBES="${NUM_PROBES:-8}"
if [[ "${NUM_PROBES}" != "8" && "${NUM_PROBES}" != "12" ]]; then
  echo "NUM_PROBES must be 8 or 12, got ${NUM_PROBES}" >&2
  exit 2
fi
QUERY_PATTERN='loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}'
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_probe${NUM_PROBES}_signed_coordinate_square/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

run_score() {
  local contraction="$1"
  local label="$2"
  local run_id="${SLURM_JOB_ID}_${label}"
  local pids=() failed=0
  echo "[score] contraction=${contraction}"
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --epochs "${JAX_EPOCHS}" --run-id "${run_id}" \
          --shard-index "${shard}" --shard-count 2 \
          --num-probes "${NUM_PROBES}" --contraction "${contraction}" \
          --query-namespace-pattern "${QUERY_PATTERN}" \
          --expected-query-probe-mode independent_gaussian
    ) >"${LOG_ROOT}/${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "${label} shard failed; inspect ${LOG_ROOT}" >&2; exit 1; }
  "${PYTHON_BIN}" "${SCORE_DRIVER}" merge \
    --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --epochs "${JAX_EPOCHS}" --run-id "${run_id}" --shard-count 2 \
    --num-probes "${NUM_PROBES}" --contraction "${contraction}" \
    --query-namespace-pattern "${QUERY_PATTERN}" \
    --expected-query-probe-mode independent_gaussian
}

echo "[phase 1/4] signed square z*abs(z)"
run_score signed_squared signed_square
echo "[phase 2/4] coordinatewise square sum_k (train_k*query_k)^2"
run_score coordinatewise_squared coordinate_square

echo "[phase 3/4] cached LDS: signed square uses sign +1"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --score-schemes "predicted_noise_jvp_signed_squared_probe${NUM_PROBES}" --prediction-sign=1

echo "[phase 4/4] cached LDS: coordinate energy uses sign -1"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --score-schemes "predicted_noise_jvp_coordinatewise_squared_probe${NUM_PROBES}" --prediction-sign=-1

"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
  --experiment "${EXPERIMENT_TAG}" --num-probes "${NUM_PROBES}" \
  --reduction signed_square --prediction-sign p1
"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
  --experiment "${EXPERIMENT_TAG}" --num-probes "${NUM_PROBES}" \
  --reduction coordinate_square --prediction-sign m1

echo "[done] ${NUM_PROBES}-probe signed-square and coordinate-square LDS complete"
