#!/usr/bin/env bash
#SBATCH -J cifar2-s3-das-attr-a
#SBATCH -o cifar2-s3-das-attr-a-%j.out
#SBATCH -e cifar2-s3-das-attr-a-%j.err
#SBATCH -p h100
#SBATCH -A CCR25021
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail
SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/02_das_attribution_chunk_stampede3.sh" ]]; then SCRIPT_DIR="${candidate}"; break; fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh" ]]; then SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}" ATTR_JOB_INDEX=0 ATTR_NUM_JOBS=3 ATTR_CHUNK_SIZE=16
exec bash "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh"
