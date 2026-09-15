#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"

"${PYTHON_BIN}" "${SHAPES_ROOT}/script/materialize_f_next_final_score_squared.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}"

JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --score-schemes f_next_final_score_squared \
  --prediction-sign=-1

"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_f_next_final_score_squared_lds.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --prediction-sign m1
