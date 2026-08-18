#!/usr/bin/env bash
#SBATCH -J x3-s3-das-eval-rtx
#SBATCH -o x3-s3-das-eval-rtx-%j.out
#SBATCH -e x3-s3-das-eval-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH -t 04:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/03_das_lds_eval_report_stampede3.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/x3/stampede3_das"; do
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
export STAMPEDE3_SLOT_BACKEND="${STAMPEDE3_SLOT_BACKEND:-local}"
export EVAL_SERIAL_SLOTS="${EVAL_SERIAL_SLOTS:-0}"
export EVAL_LOCAL_PARALLEL_SLOTS="${EVAL_LOCAL_PARALLEL_SLOTS:-4}"
export EVAL_SLOT_SHARD_COUNT="${EVAL_SLOT_SHARD_COUNT:-1}"
export LDS_EVAL_DEVICE_MODE="${LDS_EVAL_DEVICE_MODE:-gpu_then_cpu}"

exec bash "${SCRIPT_DIR}/03_das_lds_eval_report_stampede3.sh"
