#!/usr/bin/env bash
#SBATCH -J 3d-pn12-sqplay
#SBATCH -o 3d-pn12-sqplay-%j.out
#SBATCH -e 3d-pn12-sqplay-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00

set -euo pipefail

candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_predicted_noise_probe8_signed_coordinate_square_rtx_small.sh" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
BASE_SCRIPT="${REPO_ROOT}/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_predicted_noise_probe8_signed_coordinate_square_rtx_small.sh"
[[ -f "${BASE_SCRIPT}" ]] || { echo "Could not locate repository" >&2; exit 1; }
export NUM_PROBES=12
source "${BASE_SCRIPT}"
