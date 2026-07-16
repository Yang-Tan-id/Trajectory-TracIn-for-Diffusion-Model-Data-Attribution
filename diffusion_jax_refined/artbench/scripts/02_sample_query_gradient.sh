#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

ALGORITHMS_TEXT="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"
ALGORITHMS_TEXT="${ALGORITHMS_TEXT//,/ }"
SAMPLE_SEEDS_TEXT="${SAMPLE_SEEDS:-${INITIAL_SEED:-${SAMPLE_SEED:-0}}}"
SAMPLE_SEEDS_TEXT="${SAMPLE_SEEDS_TEXT//,/ }"
SAMPLE_MODEL_MODE_RAW="${SAMPLE_MODEL_MODE:-prompted_solo}"

case "${SAMPLE_MODEL_MODE_RAW}" in
  prompt|prompted|prompted_jax|prompted_solo)
    SAMPLE_MODEL_MODE_TAG="prompted_solo"
    export UNPROMPTED="${UNPROMPTED:-0}"
    export ATTRIBUTION_SAMPLE_MODEL_MODE="${ATTRIBUTION_SAMPLE_MODEL_MODE:-${SAMPLE_MODEL_MODE_TAG}}"
    ;;
  multi|prompted_multi)
    SAMPLE_MODEL_MODE_TAG="prompted_multi"
    export UNPROMPTED="${UNPROMPTED:-0}"
    export ATTRIBUTION_SAMPLE_MODEL_MODE="${ATTRIBUTION_SAMPLE_MODEL_MODE:-${SAMPLE_MODEL_MODE_TAG}}"
    ;;
  unprompted|unprompted_jax|unprompted_solo)
    SAMPLE_MODEL_MODE_TAG="unprompted_solo"
    export UNPROMPTED=1
    export UNPROMPTED_SAMPLE_MODEL_MODE="${UNPROMPTED_SAMPLE_MODEL_MODE:-${SAMPLE_MODEL_MODE_TAG}}"
    ;;
  unprompted_multi)
    SAMPLE_MODEL_MODE_TAG="unprompted_multi"
    export UNPROMPTED=1
    export UNPROMPTED_SAMPLE_MODEL_MODE="${UNPROMPTED_SAMPLE_MODEL_MODE:-${SAMPLE_MODEL_MODE_TAG}}"
    ;;
  *)
    echo "Unknown SAMPLE_MODEL_MODE=${SAMPLE_MODEL_MODE_RAW}" >&2
    echo "Expected: prompted_solo, prompted_multi, unprompted_solo, unprompted_multi, prompted, unprompted, multi" >&2
    exit 2
    ;;
esac

export SAMPLE_MODEL_MODE="${SAMPLE_MODEL_MODE_TAG}"
export SAMPLE_ROOT="${SAMPLE_ROOT:-${ROOT}/result/${EXPERIMENT_TAG:-experiment1}/sample}"

echo "Sample + query-gradient job: experiment=${EXPERIMENT_TAG:-experiment1}, model_mode=${SAMPLE_MODEL_MODE}, seeds=${SAMPLE_SEEDS_TEXT}, algorithms=${ALGORITHMS_TEXT}"
bash "${ROOT}/scripts/00_sample_for_attribution.sh"

for SAMPLE_SEED_VALUE in ${SAMPLE_SEEDS_TEXT}; do
  for ALGORITHM in ${ALGORITHMS_TEXT}; do
    echo "Computing query gradient: seed=${SAMPLE_SEED_VALUE}, algorithm=${ALGORITHM}, model_mode=${SAMPLE_MODEL_MODE}"
    (
      cd "${ROOT}/data_attribution/${ALGORITHM}"
      INITIAL_SEED="${SAMPLE_SEED_VALUE}" SAMPLE_SEED="${SAMPLE_SEED_VALUE}" "${PYTHON_BIN}" 02_query_gradient.py
    )
  done
done
