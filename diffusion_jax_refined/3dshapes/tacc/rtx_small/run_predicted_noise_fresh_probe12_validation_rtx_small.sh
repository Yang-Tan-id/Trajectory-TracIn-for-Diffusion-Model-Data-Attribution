#!/usr/bin/env bash
#SBATCH -J 3d-pn12-fresh
#SBATCH -o 3d-pn12-fresh-%j.out
#SBATCH -e 3d-pn12-fresh-%j.err
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
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; then
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# Precommitted before observing this bank's LDS. Do not tune this value.
PROBE_BANK_SEED=20260915
NUM_PROBES=12
NAMESPACE_SUFFIX="fresh_seed${PROBE_BANK_SEED}"
QUERY_PATTERN="loss_direction_residual_rms_predicted_noise_${NAMESPACE_SUFFIX}_r{probe_index}"
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_probe12_${NAMESPACE_SUFFIX}/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "[phase 1/3] generate a genuinely fresh 12-probe bank | probe_seed=${PROBE_BANK_SEED}"
for probe_index in 0 1 2 3 4 5 6 7 8 9 10 11; do
  namespace="loss_direction_residual_rms_predicted_noise_${NAMESPACE_SUFFIX}_r${probe_index}"
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
    --predicted-noise-probe-count "${NUM_PROBES}" \
    --predicted-noise-probe-seed "${PROBE_BANK_SEED}" \
    --num-snapshots 10 \
    --log-prefix "fresh_probe_${probe_index}" \
    --python-bin "${PYTHON_BIN}"
done

echo "[phase 2/3] score fresh bank with the precommitted linear reduction"
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
        --contraction final_post_square \
        --query-namespace-pattern "${QUERY_PATTERN}" \
        --expected-query-probe-mode independent_gaussian \
        --expected-query-probe-seed "${PROBE_BANK_SEED}" \
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
  --contraction final_post_square \
  --query-namespace-pattern "${QUERY_PATTERN}" \
  --expected-query-probe-mode independent_gaussian \
  --expected-query-probe-seed "${PROBE_BANK_SEED}" \
  --namespace-suffix "${NAMESPACE_SUFFIX}"

echo "[phase 3/3] cached LDS | linear sign fixed at +1 before seeing results"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --score-schemes predicted_noise_jvp_final_linear_mean_probe12_fresh_seed20260915 \
  --prediction-sign=1

"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_probe4_final_post_square_lds.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --num-probes 12 \
  --method linear \
  --prediction-sign p1 \
  --namespace-suffix "${NAMESPACE_SUFFIX}"

echo "[done] precommitted fresh 12-probe validation is complete"
