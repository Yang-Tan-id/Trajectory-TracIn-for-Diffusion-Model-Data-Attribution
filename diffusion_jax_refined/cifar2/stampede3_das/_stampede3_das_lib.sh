#!/usr/bin/env bash

stampede3_das_init() {
  if [[ -z "${REPO_ROOT:-}" ]]; then
    if [[ -n "${SCRIPT_DIR:-}" ]]; then
      REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    else
      REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
    fi
  fi
  CIFAR2_ROOT="${CIFAR2_ROOT:-${REPO_ROOT}/diffusion_jax_refined/cifar2}"
  REFINE_ROOT="${REFINE_ROOT:-${REPO_ROOT}/diffusion_jax_refined}"
  EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment_67}"
  TRAIN_SEED="${TRAIN_SEED:-67}"
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  export REPO_ROOT CIFAR2_ROOT REFINE_ROOT EXPERIMENT_TAG TRAIN_SEED PYTHON_BIN

  unset PYTHONPATH
  if [[ -n "${ENV_SETUP:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_SETUP}"
  else
    if [[ -z "${CONDA_ENV_PATH:-}" ]]; then
      if [[ -n "${SCRATCH:-}" ]]; then
        CONDA_ENV_PATH="${SCRATCH}/conda-envs/trajectory-tracin"
      else
        CONDA_ENV_PATH="${HOME}/conda-envs/trajectory-tracin"
      fi
    fi
    if [[ -n "${SCRATCH:-}" && -f "${SCRATCH}/miniforge3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_PATH}"
    elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${HOME}/miniforge3/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_PATH}"
    else
      echo "Could not find conda.sh. Set ENV_SETUP to activate the runtime." >&2
      exit 1
    fi
  fi

  [[ -d "${CIFAR2_ROOT}" ]] || {
    echo "CIFAR2 root not found: ${CIFAR2_ROOT}" >&2
    exit 1
  }
  command -v "${PYTHON_BIN}" >/dev/null || {
    echo "${PYTHON_BIN} is unavailable" >&2
    exit 1
  }
  cd "${CIFAR2_ROOT}"
}

path_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//+/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
  while [[ "${value}" == *"__"* ]]; do value="${value//__/_}"; done
  value="${value#_}"
  value="${value%_}"
  printf '%s' "${value:-unprompted}"
}

damping_tag() {
  local value="$1"
  value="${value//+/_}"
  value="${value//-/neg_}"
  value="${value//./p}"
  path_tag "${value}"
}

mode_is_unprompted() {
  [[ "$1" == unprompted_* ]]
}

log_tag() {
  local mode="$1"
  local query="${2:-unprompted}"
  local seed="$3"
  printf '%s__%s__seed_%s' "${mode}" "$(path_tag "${query}")" "${seed}"
}

wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || {
    echo "At least one task failed. Check logs above." >&2
    exit 1
  }
}

run_gpu_slot() {
  local slot="$1"
  shift
  local backend="${STAMPEDE3_SLOT_BACKEND:-ibrun}"
  if [[ "${backend}" == "ibrun" && -n "${SLURM_JOB_ID:-}" ]] && command -v ibrun >/dev/null; then
    if command -v task_affinity >/dev/null; then
      ibrun -n 1 -o "${slot}" task_affinity "$@"
    else
      ibrun -n 1 -o "${slot}" "$@"
    fi
  else
    "$@"
  fi
}

submit_job() {
  local output job_id
  output="$("$@")"
  printf '%s\n' "${output}" >&2
  job_id="$(printf '%s\n' "${output}" | awk '/^[0-9]+(;[0-9]+)?$/ {print $1}' | tail -n 1 | cut -d';' -f1)"
  if [[ -z "${job_id}" ]]; then
    echo "Could not parse job id from sbatch output above." >&2
    exit 1
  fi
  printf '%s' "${job_id}"
}
