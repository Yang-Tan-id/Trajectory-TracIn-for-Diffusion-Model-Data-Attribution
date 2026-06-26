#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
START_INDEX="${2:-0}"
END_INDEX="${3:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALGORITHM="$(basename "$PWD")"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export GPU_IDS="${GPU_IDS:-${CUDA_DEVICE}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

run_unprompted_training() {
  for TRAIN_SEED_VALUE in 0 1 2; do
    export TRAIN_SEED="${TRAIN_SEED_VALUE}"
    for SUBSET_INDEX_VALUE in $(seq "${START_INDEX}" "${END_INDEX}"); do
      export SUBSET_INDEX="${SUBSET_INDEX_VALUE}"
      python run_training.py
    done
  done
}

case "${MODE}" in
  train|train_prompted_jax)
    echo "${ALGORITHM}: prompted JAX training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/00_train_prompted_jax.sh"
    ;;
  train_unprompted)
    echo "${ALGORITHM}: unprompted diffusers training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, GPU_IDS=${GPU_IDS}"
    run_unprompted_training
    ;;
  attribution)
    python run_attribution.py
    ;;
  eval)
    python run_eval.py
    ;;
  all)
    echo "${ALGORITHM}: prompted JAX training + attribution + eval"
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/00_train_prompted_jax.sh"
    python run_attribution.py
    python run_eval.py
    ;;
  attribution_unprompted)
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/01_data_attribution_unprompted.sh"
    ;;
  eval_unprompted)
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/02_metric_counterfactual_unprompted.sh"
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/03_metric_lds_unprompted.sh"
    ;;
  all_unprompted)
    echo "${ALGORITHM}: unprompted diffusers training + attribution + eval"
    run_unprompted_training
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/01_data_attribution_unprompted.sh"
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/02_metric_counterfactual_unprompted.sh"
    ALGORITHMS="${ALGORITHM}" bash "${ROOT}/scripts/03_metric_lds_unprompted.sh"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected: train, train_prompted_jax, train_unprompted, attribution, attribution_unprompted, eval, eval_unprompted, all, all_unprompted" >&2
    exit 2
    ;;
esac
