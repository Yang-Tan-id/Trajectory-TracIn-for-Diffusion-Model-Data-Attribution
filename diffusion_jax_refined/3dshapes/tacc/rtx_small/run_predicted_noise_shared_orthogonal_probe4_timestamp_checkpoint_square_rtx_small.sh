#!/usr/bin/env bash
#SBATCH -J 3d-pn4o-tsckptsq
#SBATCH -o 3d-pn4o-tsckptsq-%j.out
#SBATCH -e 3d-pn4o-tsckptsq-%j.err
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
[[ -f "${REPO_ROOT}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"

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

NUM_PROBES=4
QUERY_PATTERN='predicted_noise_shared_orthogonal_probe4_r{probe_index}'
NAMESPACE_SUFFIX=orthogonal_shared
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_shared_orthogonal_probe4_timestamp_checkpoint_square/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "[phase 1/2] shared orthogonal probe4: per timestamp, sum 50 LR-weighted checkpoints, then square"
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
        --num-probes "${NUM_PROBES}" \
        --contraction timestamp_checkpoint_square \
        --query-namespace-pattern "${QUERY_PATTERN}" \
        --expected-query-probe-mode shared_orthogonal \
        --namespace-suffix "${NAMESPACE_SUFFIX}"
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
  --num-probes "${NUM_PROBES}" \
  --contraction timestamp_checkpoint_square \
  --query-namespace-pattern "${QUERY_PATTERN}" \
  --expected-query-probe-mode shared_orthogonal \
  --namespace-suffix "${NAMESPACE_SUFFIX}"

echo "[phase 2/2] cached LDS with squared-score sign -1"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --score-schemes predicted_noise_shared_orthogonal_probe4_timestamp_checkpoint_sum_square \
  --prediction-sign=-1

"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --num-probes 4 \
  --namespace-suffix "${NAMESPACE_SUFFIX}" \
  --prediction-sign m1

echo "[done] shared orthogonal probe4 timestamp-grouped checkpoint-square scores and LDS complete"
