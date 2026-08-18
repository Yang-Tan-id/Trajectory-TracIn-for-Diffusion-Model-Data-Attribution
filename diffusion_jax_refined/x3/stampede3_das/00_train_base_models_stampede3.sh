#!/usr/bin/env bash
#SBATCH -J x3-s3-das-train
#SBATCH -o x3-s3-das-train-%j.out
#SBATCH -e x3-s3-das-train-%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=24
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/x3/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/_stampede3_das_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_stampede3_das_lib.sh
source "${SCRIPT_DIR}/_stampede3_das_lib.sh"
stampede3_das_init

LOG_ROOT="${X3_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/00_train_base/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 00 Stampede3 DAS: train prompted/unprompted base models"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; node=1; prompted GPUs=0,1; unprompted GPUs=2,3"
echo "logs=${LOG_ROOT}"

pids=()
(
  run_gpu_slot 0 env \
    CUDA_VISIBLE_DEVICES=0,1 \
    GPU_IDS=0,1 \
    JAX_NUM_DEVICES=2 \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    TRAIN_MODE=prompted_solo \
    TRAIN_MODES=prompted_solo \
    bash scripts/script_0.sh prompted_solo
) >"${LOG_ROOT}/prompted_solo.log" 2>&1 &
pids+=("$!")

(
  run_gpu_slot 2 env \
    CUDA_VISIBLE_DEVICES=2,3 \
    GPU_IDS=2,3 \
    JAX_NUM_DEVICES=2 \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    TRAIN_MODE=unprompted_solo \
    TRAIN_MODES=unprompted_solo \
    bash scripts/script_0.sh unprompted_solo
) >"${LOG_ROOT}/unprompted_solo.log" 2>&1 &
pids+=("$!")

wait_all "${pids[@]}"
echo "Job 00 complete."
