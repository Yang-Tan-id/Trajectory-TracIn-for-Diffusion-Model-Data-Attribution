#!/usr/bin/env bash
#SBATCH -J 3d-pnfix-gsq
#SBATCH -o 3d-pnfix-gsq-%j.out
#SBATCH -e 3d-pnfix-gsq-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"
PRINT_DRIVER="${SHAPES_ROOT}/script/print_predicted_noise_fixed8_two_group_lds.py"
[[ -f "${SCORE_DRIVER}" && -f "${PRINT_DRIVER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

QUERY_PATTERN='predicted_noise_shared_orthogonal_probe4_r{probe_index}'
NAMESPACE_SUFFIX=orthogonal_extended_groupaudit
RUN_ID="${SLURM_JOB_ID}"
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_fixed8_two_group_product_square/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "[phase 1/2] square every train-query product and retain all eight probes"
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
        --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --epochs "${JAX_EPOCHS}" --run-id "${RUN_ID}" \
        --shard-index "${shard}" --shard-count 2 \
        --num-probes 8 --contraction termwise_squared_per_probe \
        --query-namespace-pattern "${QUERY_PATTERN}" \
        --expected-query-probe-mode shared_orthogonal_extended \
        --namespace-suffix "${NAMESPACE_SUFFIX}"
  ) >"${LOG_ROOT}/score_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "Score shard failed; inspect ${LOG_ROOT}" >&2; exit 1; }

echo "[phase 2/2] print product-square probes 1-4 versus 5-8 for every query"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${PRINT_DRIVER}" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --run-id "${RUN_ID}" \
  --prediction-sign -1 \
  --reduction termwise_square \
  --namespace-suffix "${NAMESPACE_SUFFIX}"

echo "[done] fixed two-group product-square per-query LDS complete"
