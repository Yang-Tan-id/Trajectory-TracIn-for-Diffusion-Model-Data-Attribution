#!/usr/bin/env bash
#SBATCH -J x3-s3-das-attr-c
#SBATCH -o x3-s3-das-attr-c-%j.out
#SBATCH -e x3-s3-das-attr-c-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail
SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/x3/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/02_das_attribution_chunk_stampede3.sh" ]]; then SCRIPT_DIR="${candidate}"; break; fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh" ]]; then SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}" ATTR_JOB_INDEX=2 ATTR_NUM_JOBS=3 ATTR_CHUNK_SIZE=16
exec bash "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh"
