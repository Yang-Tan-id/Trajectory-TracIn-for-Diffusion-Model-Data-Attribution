#!/usr/bin/env bash
#SBATCH -J 3d-q3-pconv
#SBATCH -o 3d-q3-pconv-%j.out
#SBATCH -e 3d-q3-pconv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q3_delta_weighted_probe_convergence.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
QUERY_ID="${QUERY_ID:-3}"
OUTDIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q${QUERY_ID}_delta_weighted_probe_convergence/run_${SLURM_JOB_ID}"

python "${SHAPES_ROOT}/script/analyze_q3_delta_weighted_probe_convergence.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query "${QUERY_ID}" --out-dir "${OUTDIR}"
