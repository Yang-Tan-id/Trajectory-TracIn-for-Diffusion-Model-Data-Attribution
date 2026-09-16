#!/usr/bin/env bash
#SBATCH -J 3d-pn12-sqsub
#SBATCH -o 3d-pn12-sqsub-%j.out
#SBATCH -e 3d-pn12-sqsub-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"
ANALYZER="${SHAPES_ROOT}/script/analyze_predicted_noise_probe8_all_subset_sizes.py"
[[ -f "${SCORE_DRIVER}" && -f "${ANALYZER}" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}

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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

NUM_PROBES=12
QUERY_PATTERN='loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}'
SCORE_NAMESPACE="traj_tracin_predicted_noise_jvp_termwise_squared_per_probe_probe12"
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_probe12_termwise_square_all_subsets/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "[phase 1/2] retain each probe's termwise-square trajectory score"
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
        --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --epochs "${JAX_EPOCHS}" --run-id "${SLURM_JOB_ID}" \
        --shard-index "${shard}" --shard-count 2 \
        --num-probes "${NUM_PROBES}" \
        --contraction termwise_squared_per_probe \
        --query-namespace-pattern "${QUERY_PATTERN}" \
        --expected-query-probe-mode independent_gaussian
  ) >"${LOG_ROOT}/score_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "Score shard failed; inspect ${LOG_ROOT}" >&2; exit 1; }

echo "[phase 2/2] enumerate all 4095 nonempty subsets and compute cached LDS"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${ANALYZER}" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --run-id "${SLURM_JOB_ID}" \
  --num-probes "${NUM_PROBES}" \
  --prediction-sign=-1 \
  --score-namespace "${SCORE_NAMESPACE}" \
  --analysis-label probe12_all_subset_sizes_termwise_square \
  --score-label termwise-square

echo "[done] all 12-probe termwise-square subsets evaluated"
