#!/usr/bin/env bash
#SBATCH -J 3d-q0-p3ov
#SBATCH -o 3d-q0-p3ov-%j.out
#SBATCH -e 3d-q0-p3ov-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q0_three_probe_raw_projected_overlap.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
ANALYZER="${REPO_ROOT}/diffusion_jax_refined/3dshapes/script/analyze_q0_three_probe_raw_projected_overlap.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
  python "${ANALYZER}" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" \
    --query-id 0 \
    --source-run-id "${SOURCE_RUN_ID:-3506389}" \
    --probes "${PROBES:-7,10,12}" \
    --repeat "${CROSSFIT_REPEAT:-1}" \
    --train-fold "${CROSSFIT_TRAIN_FOLD:-0}"
