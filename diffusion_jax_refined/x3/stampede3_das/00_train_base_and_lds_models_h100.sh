#!/usr/bin/env bash
#SBATCH -J x3-base-lds-h100
#SBATCH -o x3-base-lds-h100-%j.out
#SBATCH -e x3-base-lds-h100-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/x3/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/_stampede3_das_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

source "${SCRIPT_DIR}/_stampede3_das_lib.sh"
stampede3_das_init

LDS_SEEDS_TEXT="${LDS_SEEDS:-0 1 2}"
LDS_M="${LDS_M:-64}"
LDS_K="${LDS_K:-2500}"
LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-}"
LDS_MODEL_TRAIN_SEED="${LDS_MODEL_TRAIN_SEED:-${TRAIN_SEED}}"
LOG_ROOT="${X3_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/00_train_base_and_lds/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 00: x3 train base models, then LDS models"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; lds_seeds=${LDS_SEEDS_TEXT}; m=${LDS_M}; k=${LDS_K}; pct=${LDS_DATASET_PERCENTAGE:-none}"
echo "logs=${LOG_ROOT}"

base_pids=()
(
  run_gpu_slot 0 env CUDA_VISIBLE_DEVICES=0,1 GPU_IDS=0,1 JAX_NUM_DEVICES=2 \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
    TRAIN_MODE=prompted_solo TRAIN_MODES=prompted_solo \
    bash scripts/script_0.sh prompted_solo
) >"${LOG_ROOT}/base_prompted_solo.log" 2>&1 &
base_pids+=("$!")

(
  run_gpu_slot 2 env CUDA_VISIBLE_DEVICES=2,3 GPU_IDS=2,3 JAX_NUM_DEVICES=2 \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
    TRAIN_MODE=unprompted_solo TRAIN_MODES=unprompted_solo \
    bash scripts/script_0.sh unprompted_solo
) >"${LOG_ROOT}/base_unprompted_solo.log" 2>&1 &
base_pids+=("$!")

wait_all "${base_pids[@]}"
echo "[base] complete; launching LDS models"

lds_pids=()
slot=0
for mode in prompted_solo unprompted_solo; do
  for subset_seed in ${LDS_SEEDS_TEXT}; do
    gpu=$((slot % 4))
    script="scripts/03_lds_training.sh"
    if mode_is_unprompted "${mode}"; then
      script="scripts/03_lds_training_unprompted.sh"
    fi
    log="${LOG_ROOT}/lds_${mode}_subset_seed_${subset_seed}.log"
    echo "Launch LDS mode=${mode} subset_seed=${subset_seed} slot=${slot} gpu=${gpu} -> ${log}"
    (
      run_gpu_slot "${slot}" env CUDA_VISIBLE_DEVICES="${gpu}" GPU_IDS="${gpu}" JAX_NUM_DEVICES=1 \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
        SAMPLE_MODEL_MODE="${mode}" LDS_MODEL_TRAIN_SEED="${LDS_MODEL_TRAIN_SEED}" \
        LDS_M="${LDS_M}" LDS_K="${LDS_K}" LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE}" \
        LDS_SAMPLE_RANDOM_SEED="${subset_seed}" LDS_DEVICE=gpu LDS_NUM_DEVICES=1 \
        bash "${script}"
    ) >"${log}" 2>&1 &
    lds_pids+=("$!")
    slot=$((slot + 1))
  done
done

wait_all "${lds_pids[@]}"
echo "Job 00 complete."
