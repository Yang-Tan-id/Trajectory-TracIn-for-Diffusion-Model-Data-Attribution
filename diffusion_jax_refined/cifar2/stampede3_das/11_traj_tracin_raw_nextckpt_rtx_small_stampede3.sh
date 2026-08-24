#!/usr/bin/env bash
#SBATCH -J cifar2-s3-traj-proj-rtx
#SBATCH -o cifar2-s3-traj-proj-rtx-%j.out
#SBATCH -e cifar2-s3-traj-proj-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

export ATTR_NUM_SLOTS="${ATTR_NUM_SLOTS:-2}"
export GPU_PER_NODE="${GPU_PER_NODE:-2}"
export STAMPEDE3_SLOT_BACKEND="${STAMPEDE3_SLOT_BACKEND:-local}"

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/11_traj_tracin_raw_nextckpt_projected_body.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/stampede3_das" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/11_traj_tracin_raw_nextckpt_projected_body.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/11_traj_tracin_raw_nextckpt_projected_body.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

bash "${SCRIPT_DIR}/11_traj_tracin_raw_nextckpt_projected_body.sh"
