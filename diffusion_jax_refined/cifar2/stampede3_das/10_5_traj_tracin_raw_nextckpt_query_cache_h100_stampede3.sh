#!/usr/bin/env bash
#SBATCH -J cifar2-s3-qgrad-raw-h100
#SBATCH -o cifar2-s3-qgrad-raw-h100-%j.out
#SBATCH -e cifar2-s3-qgrad-raw-h100-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

export ATTR_NUM_SLOTS="${ATTR_NUM_SLOTS:-16}"
export GPU_PER_NODE="${GPU_PER_NODE:-4}"

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/10_5_traj_tracin_raw_nextckpt_query_cache_body.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/stampede3_das" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/10_5_traj_tracin_raw_nextckpt_query_cache_body.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/10_5_traj_tracin_raw_nextckpt_query_cache_body.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

bash "${SCRIPT_DIR}/10_5_traj_tracin_raw_nextckpt_query_cache_body.sh"
