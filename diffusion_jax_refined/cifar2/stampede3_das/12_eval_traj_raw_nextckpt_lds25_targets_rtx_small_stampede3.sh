#!/usr/bin/env bash
#SBATCH -J cifar2-s3-eval-raw-next-rtx
#SBATCH -o cifar2-s3-eval-raw-next-rtx-%j.out
#SBATCH -e cifar2-s3-eval-raw-next-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/12_eval_traj_raw_nextckpt_lds25_targets_stampede3.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/stampede3_das" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/12_eval_traj_raw_nextckpt_lds25_targets_stampede3.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/12_eval_traj_raw_nextckpt_lds25_targets_stampede3.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}"
export STAMPEDE3_SLOT_BACKEND="${STAMPEDE3_SLOT_BACKEND:-local}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment_67}"
export TRAIN_SEED="${TRAIN_SEED:-67}"
export GPU_PER_NODE="${GPU_PER_NODE:-2}"
export EVAL_SERIAL_SLOTS="${EVAL_SERIAL_SLOTS:-0}"
export EVAL_LOCAL_PARALLEL_SLOTS="${EVAL_LOCAL_PARALLEL_SLOTS:-4}"
export EVAL_SLOT_SHARD_COUNT="${EVAL_SLOT_SHARD_COUNT:-1}"
export LDS_EVAL_DEVICE_MODE="${LDS_EVAL_DEVICE_MODE:-gpu_then_cpu}"

bash "${SCRIPT_DIR}/12_eval_traj_raw_nextckpt_lds25_targets_stampede3.sh"
