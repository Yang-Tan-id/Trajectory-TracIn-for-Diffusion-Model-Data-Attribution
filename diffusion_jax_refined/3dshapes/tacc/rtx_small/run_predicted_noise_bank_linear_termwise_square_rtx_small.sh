#!/usr/bin/env bash
#SBATCH -J 3d-pn-lin-termsq
#SBATCH -o 3d-pn-lin-termsq-%j.out
#SBATCH -e 3d-pn-lin-termsq-%j.err
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

case "${PROBE_BANK:-}" in
  independent12)
    NUM_PROBES=12
    QUERY_PATTERN='loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}'
    EXPECTED_MODE=independent_gaussian
    NAMESPACE_SUFFIX=""
    LINEAR_SCHEME=predicted_noise_jvp_signed_probe12
    SQUARE_SCHEME=predicted_noise_jvp_l2_squared_probe12
    ;;
  fixed8)
    NUM_PROBES=8
    QUERY_PATTERN='predicted_noise_shared_orthogonal_probe4_r{probe_index}'
    EXPECTED_MODE=shared_orthogonal_extended
    NAMESPACE_SUFFIX=orthogonal_extended
    LINEAR_SCHEME=predicted_noise_shared_orthogonal_probe8_linear
    SQUARE_SCHEME=predicted_noise_shared_orthogonal_probe8_termwise_square
    ;;
  *)
    echo "Set PROBE_BANK=independent12 or PROBE_BANK=fixed8" >&2
    exit 2
    ;;
esac

LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_${PROBE_BANK}_linear_termwise_square/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

run_score() {
  local contraction="$1" label="$2" run_id="${SLURM_JOB_ID}_${label}"
  local -a suffix_args=()
  [[ -z "${NAMESPACE_SUFFIX}" ]] || suffix_args=(--namespace-suffix "${NAMESPACE_SUFFIX}")
  local pids=() failed=0
  echo "[score] bank=${PROBE_BANK} contraction=${contraction}"
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --epochs "${JAX_EPOCHS}" --run-id "${run_id}" \
          --shard-index "${shard}" --shard-count 2 \
          --num-probes "${NUM_PROBES}" --contraction "${contraction}" \
          --query-namespace-pattern "${QUERY_PATTERN}" \
          --expected-query-probe-mode "${EXPECTED_MODE}" "${suffix_args[@]}"
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
    --expected-query-probe-mode "${EXPECTED_MODE}" "${suffix_args[@]}"
}

echo "[phase 1/3] linear score"
run_score signed linear
echo "[phase 2/3] square each checkpoint/timestamp/probe product, then sum"
run_score squared termwise_square

echo "[phase 3/3] cached LDS and per-query tables"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --score-schemes "${LINEAR_SCHEME}" --prediction-sign=1
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --score-schemes "${SQUARE_SCHEME}" --prediction-sign=-1

print_args=(--experiment "${EXPERIMENT_TAG}" --num-probes "${NUM_PROBES}")
[[ -z "${NAMESPACE_SUFFIX}" ]] || print_args+=(--namespace-suffix "${NAMESPACE_SUFFIX}")
"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
  "${print_args[@]}" --reduction linear --prediction-sign p1
"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
  "${print_args[@]}" --reduction termwise_square --prediction-sign m1
echo "[done] ${PROBE_BANK} linear and termwise-square scores/LDS complete"
