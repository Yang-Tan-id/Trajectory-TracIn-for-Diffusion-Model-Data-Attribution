#!/usr/bin/env bash
#SBATCH -J 3d-pn-extsign
#SBATCH -o 3d-pn-extsign-%j.out
#SBATCH -e 3d-pn-extsign-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_predicted_noise_extreme_probe_term_signs.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
ANALYZER="${REPO_ROOT}/diffusion_jax_refined/3dshapes/script/analyze_predicted_noise_extreme_probe_term_signs.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export SOURCE_RUN_ID="${SOURCE_RUN_ID:-3506389}"

python "${ANALYZER}" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --source-run-id "${SOURCE_RUN_ID}"
