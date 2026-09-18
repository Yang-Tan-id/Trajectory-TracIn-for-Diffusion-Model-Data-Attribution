#!/usr/bin/env bash
#SBATCH -J 3d-replay4ep
#SBATCH -o 3d-replay4ep-%j.out
#SBATCH -e 3d-replay4ep-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail

candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/replay_exact_training_interval.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
START_EPOCH="${START_EPOCH:-40}"
END_EPOCH="${END_EPOCH:-44}"
OUTDIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/exact_training_interval_replay/run_${SLURM_JOB_ID}_epoch_${START_EPOCH}_${END_EPOCH}"

python "${SHAPES_ROOT}/script/replay_exact_training_interval.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --start-epoch "${START_EPOCH}" \
  --end-epoch "${END_EPOCH}" \
  --out-dir "${OUTDIR}"

echo "[done] exact replay validation: ${OUTDIR}"
