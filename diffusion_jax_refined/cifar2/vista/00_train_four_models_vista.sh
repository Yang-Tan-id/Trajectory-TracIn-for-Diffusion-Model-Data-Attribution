#!/usr/bin/env bash
#SBATCH --job-name=cifar2-train-base
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=cifar2-train-4models-%j.out
#SBATCH --error=cifar2-train-4models-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_vista_pipeline_lib.sh
source "${SCRIPT_DIR}/_vista_pipeline_lib.sh"
vista_init

MODEL_MODES=(prompted_solo unprompted_solo)
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/00_train_four_models/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 00: train base checkpoint families"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; modes=${MODEL_MODES[*]}"
echo "prompted_solo/prompted_multi share prompted_jax checkpoints; unprompted_solo/unprompted_multi share unprompted_jax checkpoints."
echo "logs=${LOG_ROOT}"

pids=()
slot=0
for mode in "${MODEL_MODES[@]}"; do
  log="${LOG_ROOT}/${mode}.log"
  echo "Launch training mode=${mode} slot=${slot} -> ${log}"
  (
    run_slot "${slot}" env \
      CUDA_VISIBLE_DEVICES=0 \
      GPU_IDS=0 \
      JAX_NUM_DEVICES=1 \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      TRAIN_MODE="${mode}" \
      TRAIN_MODES="${mode}" \
      bash scripts/script_0.sh "${mode}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"
echo "Job 00 complete."
