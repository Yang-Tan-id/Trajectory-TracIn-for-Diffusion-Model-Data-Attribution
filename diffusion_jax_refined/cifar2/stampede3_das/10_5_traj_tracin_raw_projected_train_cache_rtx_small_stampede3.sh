#!/usr/bin/env bash
#SBATCH -J cifar2-s3-projtrain-raw-rtx
#SBATCH -o cifar2-s3-projtrain-raw-rtx-%j.out
#SBATCH -e cifar2-s3-projtrain-raw-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/10_5_traj_tracin_raw_projected_train_cache_h100_stampede3.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/stampede3_das" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/10_5_traj_tracin_raw_projected_train_cache_h100_stampede3.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/10_5_traj_tracin_raw_projected_train_cache_h100_stampede3.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export GPU_SLOTS="${GPU_SLOTS:-2}"
export GPU_PER_NODE="${GPU_PER_NODE:-2}"
export PROJECTED_TRAIN_SHARD_RANGES="${PROJECTED_TRAIN_SHARD_RANGES:-1-5000 5001-10000}"

bash "${SCRIPT_DIR}/10_5_traj_tracin_raw_projected_train_cache_h100_stampede3.sh"
