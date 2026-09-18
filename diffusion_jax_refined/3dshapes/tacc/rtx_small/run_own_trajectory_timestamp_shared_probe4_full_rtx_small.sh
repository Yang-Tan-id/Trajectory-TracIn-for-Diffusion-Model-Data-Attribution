#!/usr/bin/env bash
#SBATCH -J 3d-pn4-own-ts
#SBATCH -o 3d-pn4-own-ts-%j.out
#SBATCH -e 3d-pn4-own-ts-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_reference_timestamp_shared_probe4_full_rtx_small.sh" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
BASE_RUNNER="${REPO_ROOT}/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_reference_timestamp_shared_probe4_full_rtx_small.sh"
[[ -f "${BASE_RUNNER}" ]] || { echo "Could not locate repository runner" >&2; exit 1; }
export TRAJECTORY_MODE=own
exec bash "${BASE_RUNNER}"
