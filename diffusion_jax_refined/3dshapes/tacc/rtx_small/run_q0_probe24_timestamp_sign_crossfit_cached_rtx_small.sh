#!/usr/bin/env bash
#SBATCH -J 3d-q0-tscv
#SBATCH -o 3d-q0-tscv-%j.out
#SBATCH -e 3d-q0-tscv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q0_probe24_timestamp_sign_crossfit.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
ANALYZER="${REPO_ROOT}/diffusion_jax_refined/3dshapes/script/analyze_q0_probe24_timestamp_sign_crossfit.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1

python "${ANALYZER}" \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --source-run-id "${SOURCE_RUN_ID:-3506389}" \
  --query-id 0 \
  --repeats "${CROSSFIT_REPEATS:-20}" \
  --random-seed "${CROSSFIT_SEED:-20260916}"
