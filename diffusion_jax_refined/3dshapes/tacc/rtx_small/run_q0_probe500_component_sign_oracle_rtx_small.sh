#!/usr/bin/env bash
#SBATCH -J 3d-q0-500sgn
#SBATCH -o 3d-q0-500sgn-%j.out
#SBATCH -e 3d-q0-500sgn-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q0_probe500_component_sign_oracle.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
ANALYZER="${REPO_ROOT}/diffusion_jax_refined/3dshapes/script/analyze_q0_probe500_component_sign_oracle.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RESULT_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes/result/${EXPERIMENT_TAG:-experiment1}"
OUT_DIR="${RESULT_ROOT}/eval/q0_probe500_component_sign_oracle/source_run_${SOURCE_RUN_ID:-3506389}"
mkdir -p "${OUT_DIR}"

echo "[phase 1/2] materialize Q0 per-probe predictions for all 500 components"
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "${ANALYZER}" shard \
      --experiment "${EXPERIMENT_TAG:-experiment1}" \
      --train-seed "${TRAIN_SEED:-42}" \
      --query-id 0 \
      --source-run-id "${SOURCE_RUN_ID:-3506389}" \
      --shard-count 2 \
      --shard-index "${shard}" \
      --out-dir "${OUT_DIR}"
  ) >"${OUT_DIR}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "Score shard failed; inspect ${OUT_DIR}/shard_{0,1}.log" >&2
  exit 1
fi

echo "[phase 2/2] multi-start 500-sign coordinate oracle and two-fold heldout audit"
python "${ANALYZER}" merge-optimize \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-id 0 \
  --source-run-id "${SOURCE_RUN_ID:-3506389}" \
  --shard-count 2 \
  --random-restarts "${RANDOM_RESTARTS:-2}" \
  --max-steps "${MAX_STEPS:-250}" \
  --out-dir "${OUT_DIR}"
