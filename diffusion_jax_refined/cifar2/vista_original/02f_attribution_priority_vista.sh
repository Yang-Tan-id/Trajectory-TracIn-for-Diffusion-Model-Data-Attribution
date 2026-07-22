#!/usr/bin/env bash
#SBATCH --job-name=cifar2-orig-attr-5
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar2-orig-attr-5-%j.out
#SBATCH --error=cifar2-orig-attr-5-%j.err
set -euo pipefail
export ATTR_JOB_INDEX=5
SCRIPT_DIR="${VISTA_ORIGINAL_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_attribution_chunk_vista.sh" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/vista_original"; do
    if [[ -n "${candidate}" && -f "${candidate}/02_attribution_chunk_vista.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/02_attribution_chunk_vista.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
exec bash "${SCRIPT_DIR}/02_attribution_chunk_vista.sh"
