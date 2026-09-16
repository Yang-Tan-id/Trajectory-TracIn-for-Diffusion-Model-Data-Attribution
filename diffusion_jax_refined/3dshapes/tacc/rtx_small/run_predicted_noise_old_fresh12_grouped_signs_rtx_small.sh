#!/usr/bin/env bash
#SBATCH -J 3d-pn12-gsign
#SBATCH -o 3d-pn12-gsign-%j.out
#SBATCH -e 3d-pn12-gsign-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_predicted_noise_old_fresh_grouped_signs.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
ANALYZER="${SHAPES_ROOT}/script/analyze_predicted_noise_old_fresh_grouped_signs.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export QUERY_IDS="${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}"

OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/predicted_noise_old_fresh12_grouped_signs/run_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

echo "[phase 1/2] accumulate 10 timestamp groups and 50 checkpoint groups"
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "${ANALYZER}" shard \
        --experiment "${EXPERIMENT_TAG}" \
        --train-seed "${TRAIN_SEED}" \
        --query-ids "${QUERY_IDS}" \
        --shard-index "${shard}" \
        --shard-count 2 \
        --out-dir "${OUT_DIR}"
  ) >"${OUT_DIR}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "Grouped-sign shard failed; inspect ${OUT_DIR}" >&2; exit 1; }

echo "[phase 2/2] merge, validate final score, and summarize sign distributions"
JAX_PLATFORMS=cpu python "${ANALYZER}" merge \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids "${QUERY_IDS}" \
  --shard-count 2 \
  --out-dir "${OUT_DIR}"

echo "[done] old12/fresh12 grouped-sign analysis complete"
