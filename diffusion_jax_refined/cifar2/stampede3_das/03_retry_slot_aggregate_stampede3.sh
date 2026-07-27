#!/usr/bin/env bash
#SBATCH -J cifar2-s3-das-eval-retry
#SBATCH -o cifar2-s3-das-eval-retry-%j.out
#SBATCH -e cifar2-s3-das-eval-retry-%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/03_das_lds_eval_report_stampede3.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/03_das_lds_eval_report_stampede3.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/03_das_lds_eval_report_stampede3.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}"
export EVAL_SLOT_ONLY="${EVAL_SLOT_ONLY:-3}"
export LDS_DEVICE="${LDS_DEVICE:-cpu}"

exec bash "${SCRIPT_DIR}/03_das_lds_eval_report_stampede3.sh"
