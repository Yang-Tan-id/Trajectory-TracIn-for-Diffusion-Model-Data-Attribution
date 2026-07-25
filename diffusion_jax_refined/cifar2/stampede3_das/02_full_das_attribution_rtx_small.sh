#!/usr/bin/env bash
#SBATCH -J cifar2-s3-das-full-rtx
#SBATCH -o cifar2-s3-das-full-rtx-%j.out
#SBATCH -e cifar2-s3-das-full-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/02_das_attribution_chunk_stampede3.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}"
export STAMPEDE3_SLOT_BACKEND="${STAMPEDE3_SLOT_BACKEND:-local}"
export ATTR_JOB_INDEX="${ATTR_JOB_INDEX:-0}"
export ATTR_NUM_JOBS="${ATTR_NUM_JOBS:-12}"
export ATTR_CHUNK_SIZE="${ATTR_CHUNK_SIZE:-4}"
export DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000}"

unset SCORE_INDEX_RANGES ATTRIBUTION_RANGES DAS_PROJ_DIM DAS_TIMESTEPS DAS_NUM_MC_NOISE

exec bash "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh"
