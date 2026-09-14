#!/usr/bin/env bash
#SBATCH -J 3d-das-n10l
#SBATCH -o 3d-das-n10l-%j.out
#SBATCH -e 3d-das-n10l-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
candidate="$(cd "${candidate}" && pwd)"
while [[ "${candidate}" != "/" ]]; do
  [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_das_lds_cached.py" ]] && break
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
[[ -f "${REPO_ROOT}/diffusion_jax_refined/3dshapes/script/run_das_lds_cached.py" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

cd "${REPO_ROOT}/diffusion_jax_refined/3dshapes"
JAX_PLATFORMS=cpu python script/run_das_lds_cached.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --artifact-namespace mc_normalized10x10
