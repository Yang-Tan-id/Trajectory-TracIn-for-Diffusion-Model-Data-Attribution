#!/usr/bin/env bash
#SBATCH --job-name=cifar2-orig-lds
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar2-orig-lds-%j.out
#SBATCH --error=cifar2-orig-lds-%j.err

set -euo pipefail

SCRIPT_DIR="${VISTA_ORIGINAL_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_original_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/vista_original"; do
    if [[ -n "${candidate}" && -f "${candidate}/_vista_original_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_original_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_vista_original_lib.sh
source "${SCRIPT_DIR}/_vista_original_lib.sh"
vista_original_init

MODEL_MODES=(prompted_solo unprompted_solo)
LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s ' ' 0 7)}"
LDS_M="${LDS_M:-64}"
LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"
LDS_K="${LDS_K:-5000}"
LDS_MODEL_TRAIN_SEED="${LDS_MODEL_TRAIN_SEED:-${TRAIN_SEED}}"
export LDS_M LDS_DATASET_PERCENTAGE LDS_K LDS_MODEL_TRAIN_SEED
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_original_logs/01_train_lds_models/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 01: train LDS models"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; lds_train_seed=${LDS_MODEL_TRAIN_SEED}"
echo "m=${LDS_M}; dataset_percentage=${LDS_DATASET_PERCENTAGE}; subset_seeds=${LDS_SEEDS_TEXT}; modes=${MODEL_MODES[*]}"
echo "logs=${LOG_ROOT}"

pids=()
slot=0
for mode in "${MODEL_MODES[@]}"; do
  for subset_seed in ${LDS_SEEDS_TEXT}; do
    log="${LOG_ROOT}/${mode}_subset_seed_${subset_seed}.log"
    echo "Launch LDS mode=${mode} subset_seed=${subset_seed} slot=${slot} -> ${log}"
    script="scripts/03_lds_training.sh"
    if mode_is_unprompted "${mode}"; then
      script="scripts/03_lds_training_unprompted.sh"
    fi
    (
      run_slot "${slot}" env \
        CUDA_VISIBLE_DEVICES=0 \
        GPU_IDS=0 \
        JAX_NUM_DEVICES=1 \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
        TRAIN_SEED="${TRAIN_SEED}" \
        SAMPLE_MODEL_MODE="${mode}" \
        LDS_MODEL_TRAIN_SEED="${LDS_MODEL_TRAIN_SEED}" \
        LDS_M="${LDS_M}" \
        LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE}" \
        LDS_SAMPLE_RANDOM_SEED="${subset_seed}" \
        LDS_DEVICE=gpu \
        LDS_NUM_DEVICES=1 \
        bash "${script}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
    slot=$((slot + 1))
  done
done

wait_all "${pids[@]}"
echo "Job 01 complete."
