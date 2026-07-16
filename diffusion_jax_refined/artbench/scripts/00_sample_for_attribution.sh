#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export SAMPLE_MODEL_MODE="${SAMPLE_MODEL_MODE:-prompted_solo}"
export SAMPLE_ROOT="${SAMPLE_ROOT:-${ROOT}/result/${EXPERIMENT_TAG:-experiment1}/sample}"

case "${SAMPLE_MODEL_MODE}" in
  unprompted|unprompted_jax|unprompted_solo|unprompted_multi)
    export UNPROMPTED=1
    export UNPROMPTED_SAMPLE_MODEL_MODE="${UNPROMPTED_SAMPLE_MODEL_MODE:-${SAMPLE_MODEL_MODE}}"
    ;;
  prompt|prompted|prompted_jax|prompted_solo|prompted_multi|multi)
    export UNPROMPTED="${UNPROMPTED:-0}"
    export ATTRIBUTION_SAMPLE_MODEL_MODE="${ATTRIBUTION_SAMPLE_MODEL_MODE:-${SAMPLE_MODEL_MODE}}"
    ;;
  *)
    echo "Unknown SAMPLE_MODEL_MODE=${SAMPLE_MODEL_MODE}" >&2
    echo "Expected: prompted_solo, prompted_multi, unprompted_solo, unprompted_multi, prompted, unprompted, multi" >&2
    exit 2
    ;;
esac

echo "Sampling data-attribution trajectory: experiment=${EXPERIMENT_TAG:-experiment1}, model_mode=${SAMPLE_MODEL_MODE}, sample_root=${SAMPLE_ROOT}, cuda=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" "${ROOT}/sampling/run_sampling.py"
