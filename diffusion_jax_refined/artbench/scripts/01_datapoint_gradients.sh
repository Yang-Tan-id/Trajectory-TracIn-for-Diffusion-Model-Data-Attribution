#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS_TEXT="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"
ALGORITHMS_TEXT="${ALGORITHMS_TEXT//,/ }"
TRAIN_MODES_TEXT="${TRAIN_MODES:-${TRAIN_MODE:-}}"
TRAIN_MODES_TEXT="${TRAIN_MODES_TEXT//,/ }"
GRADIENT_MODES_TEXT="${DATAPOINT_GRADIENT_MODES:-${GRADIENT_MODES:-prompted}}"
GRADIENT_MODES_TEXT="${GRADIENT_MODES_TEXT//,/ }"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

model_mode_label() {
  local kind="$1"
  local mode
  for mode in ${TRAIN_MODES_TEXT}; do
    case "${kind}:${mode}" in
      prompted:prompted_solo|prompted:prompted_multi|unprompted:unprompted_solo|unprompted:unprompted_multi)
        printf '%s\n' "${mode}"
        return
        ;;
    esac
  done
  if [[ "${kind}" == "prompted" ]]; then
    printf '%s\n' "${DATAPOINT_PROMPTED_MODEL_MODE:-prompted_solo}"
  else
    printf '%s\n' "${DATAPOINT_UNPROMPTED_MODEL_MODE:-unprompted_solo}"
  fi
}

run_gradient_mode() {
  local kind="$1"
  local mode_label
  local gradient_root
  mode_label="$(model_mode_label "${kind}")"
  gradient_root="${ROOT}/result/${EXPERIMENT_TAG:-experiment1}/model/${mode_label}/seed_${TRAIN_SEED:-42}_train_gradient"
  mkdir -p "${gradient_root}"
  echo "stage-1 root=${gradient_root}"
  for ALGORITHM in ${ALGORITHMS_TEXT}; do
    echo "Running stage-1 train datapoint gradient: kind=${kind}, algorithm=${ALGORITHM}"
    (
      cd "${ROOT}/data_attribution/${ALGORITHM}"
      DATAPOINT_MODEL_MODE="${mode_label}" "${PYTHON_BIN}" 01_train_datapoint_gradient.py
    )
  done
}

if [[ -n "${TRAIN_MODES_TEXT}" ]]; then
  echo "Running optional training modes: ${TRAIN_MODES_TEXT}"
  bash "${ROOT}/scripts/script_0.sh" ${TRAIN_MODES_TEXT}
fi

for MODE in ${GRADIENT_MODES_TEXT}; do
  case "${MODE}" in
    prompted|prompted_jax)
      echo "Computing prompted datapoint gradients/features via attribution engines: ${ALGORITHMS_TEXT}"
      run_gradient_mode prompted
      ;;
    unprompted|unprompted_jax)
      echo "Computing unprompted datapoint gradients/features via attribution engines: ${ALGORITHMS_TEXT}"
      run_gradient_mode unprompted
      ;;
    both|all)
      echo "Computing prompted and unprompted datapoint gradients/features: ${ALGORITHMS_TEXT}"
      run_gradient_mode prompted
      run_gradient_mode unprompted
      ;;
    *)
      echo "Unknown DATAPOINT_GRADIENT_MODES entry: ${MODE}" >&2
      echo "Expected: prompted, unprompted, both" >&2
      exit 2
      ;;
  esac
done
