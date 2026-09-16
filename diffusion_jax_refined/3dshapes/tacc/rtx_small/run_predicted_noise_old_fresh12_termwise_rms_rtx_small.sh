#!/usr/bin/env bash
#SBATCH -J 3d-pn12-2rms
#SBATCH -o 3d-pn12-2rms-%j.out
#SBATCH -e 3d-pn12-2rms-%j.err
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

NUM_PROBES=12
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_old_fresh12_termwise_rms/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

score_bank() {
  local label="$1"
  local query_pattern="$2"
  local suffix="$3"
  local expected_seed="$4"
  local extra_args=()
  if [[ -n "${suffix}" ]]; then
    extra_args+=(--namespace-suffix "${suffix}")
  fi
  if [[ -n "${expected_seed}" ]]; then
    extra_args+=(--expected-query-probe-seed "${expected_seed}")
  fi

  echo "[score ${label}] per term: sqrt(mean 12 probe product squares); LR outside root"
  local pids=()
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --epochs "${JAX_EPOCHS}" --run-id "${SLURM_JOB_ID}" \
          --shard-index "${shard}" --shard-count 2 \
          --num-probes "${NUM_PROBES}" --contraction rms \
          --query-namespace-pattern "${query_pattern}" \
          --expected-query-probe-mode independent_gaussian \
          "${extra_args[@]}"
    ) >"${LOG_ROOT}/${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "${label} score shard failed; inspect ${LOG_ROOT}" >&2; exit 1; }

  "${PYTHON_BIN}" "${SCORE_DRIVER}" merge \
    --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --epochs "${JAX_EPOCHS}" --run-id "${SLURM_JOB_ID}" --shard-count 2 \
    --num-probes "${NUM_PROBES}" --contraction rms \
    --query-namespace-pattern "${query_pattern}" \
    --expected-query-probe-mode independent_gaussian \
    "${extra_args[@]}"
}

score_bank \
  old12 \
  'loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}' \
  '' \
  ''

score_bank \
  fresh12 \
  'loss_direction_residual_rms_predicted_noise_fresh_seed20260915_r{probe_index}' \
  'fresh_seed20260915' \
  '20260915'

echo "[LDS] old12 and fresh12 termwise RMS; both prediction signs"
for sign in 1 -1; do
  JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --score-schemes \
      predicted_noise_jvp_rms_probe12,predicted_noise_jvp_rms_probe12_fresh_seed20260915 \
    --prediction-sign="${sign}"
done

for bank in old12 fresh12; do
  suffix_args=()
  [[ "${bank}" == fresh12 ]] && suffix_args+=(--namespace-suffix fresh_seed20260915)
  for sign in p1 m1; do
    "${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
      --experiment "${EXPERIMENT_TAG}" --num-probes 12 \
      --reduction rms --prediction-sign "${sign}" \
      "${suffix_args[@]}"
  done
done

echo "[done] old12 and fresh12 termwise probe-RMS LDS complete"
