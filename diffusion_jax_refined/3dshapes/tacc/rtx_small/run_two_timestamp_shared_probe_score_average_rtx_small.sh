#!/usr/bin/env bash
#SBATCH -J 3d-pn2-avg
#SBATCH -o 3d-pn2-avg-%j.out
#SBATCH -e 3d-pn2-avg-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_two_timestamp_shared_probe_score_average.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/two_timestamp_shared_probe_score_average/run_${SLURM_JOB_ID}"

JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_two_timestamp_shared_probe_score_average.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --out-dir "${OUT_DIR}"

echo "[done] two-probe score average: ${OUT_DIR}"
