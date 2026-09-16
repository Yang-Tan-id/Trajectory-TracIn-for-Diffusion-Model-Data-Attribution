#!/usr/bin/env bash
#SBATCH -J 3d-pn24-winner
#SBATCH -o 3d-pn24-winner-%j.out
#SBATCH -e 3d-pn24-winner-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_predicted_noise_probe24_term_winners.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
ANALYZER="${SHAPES_ROOT}/script/analyze_predicted_noise_probe24_term_winners.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export QUERY_IDS="${QUERY_IDS:-2,3}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/probe24_term_winners/run_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

echo "[phase 1/2] Q=${QUERY_IDS}: evaluate all 24 probes independently at each of 490 terms"
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      "${PYTHON_BIN}" "${ANALYZER}" analyze-shard \
        --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --query-ids "${QUERY_IDS}" --run-id "${SLURM_JOB_ID}" \
        --shard-index "${shard}" --shard-count 2 --out-dir "${OUT_DIR}"
  ) >"${OUT_DIR}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "Analysis shard failed; inspect ${OUT_DIR}" >&2; exit 1; }

echo "[phase 2/2] merge term winners and summarize next-checkpoint/train alignment"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${ANALYZER}" merge \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query-ids "${QUERY_IDS}" --run-id "${SLURM_JOB_ID}" \
  --shard-count 2 --out-dir "${OUT_DIR}"

echo "[done] 24-probe per-term winner analysis complete"
