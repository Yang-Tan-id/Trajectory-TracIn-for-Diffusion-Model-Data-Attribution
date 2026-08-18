#!/usr/bin/env bash
set -euo pipefail

echo "[sample] sample_mode=${SAMPLE_MODEL_MODE} score_mode=${ATTRIBUTION_SCORE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED}"
if [[ -f "${SAMPLE_DONE_FILE}" ]]; then
  echo "[sample] reuse existing ${SAMPLE_DONE_FILE}"
elif mkdir "${SAMPLE_LOCK_DIR}" 2>/dev/null; then
  trap 'rmdir "${SAMPLE_LOCK_DIR}" 2>/dev/null || true' EXIT
  if [[ -f "${SAMPLE_DONE_FILE}" ]]; then
    echo "[sample] reuse existing ${SAMPLE_DONE_FILE}"
  else
    echo "[sample] generating ${SAMPLE_DONE_FILE}"
    bash scripts/00_sample_for_attribution.sh
  fi
  rmdir "${SAMPLE_LOCK_DIR}" 2>/dev/null || true
  trap - EXIT
else
  echo "[sample] waiting for locked sample ${SAMPLE_DONE_FILE}"
  waited=0
  while [[ ! -f "${SAMPLE_DONE_FILE}" ]]; do
    if (( waited >= SAMPLE_LOCK_WAIT_SECONDS )); then
      echo "Timed out waiting for sample ${SAMPLE_DONE_FILE}" >&2
      exit 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "[sample] reuse after wait ${SAMPLE_DONE_FILE}"
fi

echo "[original-attribution] algorithm=${ALGORITHM} score_mode=${ATTRIBUTION_SCORE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED}"
"${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py" "${X3_ROOT}/data_attribution/${ALGORITHM}/CONFIG.py"
