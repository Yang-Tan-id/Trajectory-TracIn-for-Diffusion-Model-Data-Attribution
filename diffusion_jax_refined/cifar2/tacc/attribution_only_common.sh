#!/usr/bin/env bash

# Shared implementation for attribution-only Slurm wrappers.
# This file intentionally has no #SBATCH lines.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-${QUERY_OBJECTIVE:-trajectory_noise_squared_deviation}}"
ATTR_ONLY_LABEL="${ATTR_ONLY_LABEL:-manual}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/attr_only_${ATTR_ONLY_LABEL}_${ATTR_TASK_START}_${ATTR_TASK_END}_${SLURM_JOB_ID:-manual}"

QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")
ATTR_ALGORITHMS="${ATTR_ALGORITHMS:-traj_tracin das}"
read -r -a ATTR_ALGORITHM_LIST <<<"${ATTR_ALGORITHMS}"

unset PYTHONPATH
if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-${SCRATCH}/conda-envs/trajectory-tracin}"
  # shellcheck disable=SC1090
  source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
[[ -d "${CIFAR2_ROOT}" ]] || {
  echo "CIFAR2 root not found: ${CIFAR2_ROOT}" >&2
  echo "Submit this job from the repository root or set REPO_ROOT explicitly." >&2
  exit 1
}
command -v python >/dev/null || { echo "python is unavailable" >&2; exit 1; }
mkdir -p "${LOG_ROOT}"
cd "${CIFAR2_ROOT}"

query_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//[^A-Za-z0-9._-]/_}"
  printf '%s' "${value}"
}

path_tag() {
  local value="$1"
  value="${value//,/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
  while [[ "${value}" == *"__"* ]]; do
    value="${value//__/_}"
  done
  value="${value#_}"
  value="${value%_}"
  printf '%s' "${value}"
}

traj_algorithm_tag() {
  if [[ "${TRAJ_QUERY_OBJECTIVE}" == "trajectory_noise_squared_deviation" ]]; then
    printf '%s' "traj_tracin"
  else
    local value="${TRAJ_QUERY_OBJECTIVE}"
    value="${value//[^A-Za-z0-9._-]/_}"
    while [[ "${value}" == *"__"* ]]; do
      value="${value//__/_}"
    done
    value="${value#_}"
    value="${value%_}"
    printf 'traj_tracin_%s' "${value}"
  fi
}

sample_dir_for_query() {
  local query="$1"
  local tag
  tag="$(path_tag "${query}")"
  printf '%s/result/%s/eval/sampling/cifar/prompt_%s/model_prompted_jax__ckpt_seed_42_epoch_0200/seed_%06d' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${tag}" "${INITIAL_SEED}"
}

attribution_root_for_query() {
  printf '%s/result/%s/attribution_score/query_%s/initial_seed_%s' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "$1")" "${INITIAL_SEED}"
}

validate_sample() {
  local query="$1"
  local sample_dir
  sample_dir="$(sample_dir_for_query "${query}")"
  python - "${sample_dir}" "${query}" "${INITIAL_SEED}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

sample_dir = Path(sys.argv[1])
expected_prompt = sys.argv[2]
expected_seed = int(sys.argv[3])
required = ["trajectory_xt.npy", "trajectory_t.npy", "final_state.npy", "seed_info.json"]
missing = [name for name in required if not (sample_dir / name).is_file()]
manifest = sample_dir.parent / "manifest.json"
if not manifest.is_file():
    missing.append(str(manifest))
if missing:
    raise FileNotFoundError(f"Invalid/incomplete sample {sample_dir}; missing {missing}")

seed_info = json.loads((sample_dir / "seed_info.json").read_text())
manifest_payload = json.loads(manifest.read_text())
if int(seed_info.get("seed", -1)) != expected_seed:
    raise ValueError(f"{sample_dir}: seed_info seed={seed_info.get('seed')} expected={expected_seed}")
if expected_prompt != str(seed_info.get("prompt")):
    raise ValueError(f"{sample_dir}: seed_info prompt={seed_info.get('prompt')!r} expected={expected_prompt!r}")
if expected_prompt != str(manifest_payload.get("prompt")):
    raise ValueError(f"{manifest}: manifest prompt={manifest_payload.get('prompt')!r} expected={expected_prompt!r}")
if expected_seed not in [int(x) for x in manifest_payload.get("seeds", [])]:
    raise ValueError(f"{manifest}: expected seed {expected_seed} not in manifest seeds={manifest_payload.get('seeds')}")

trajectory = np.load(sample_dir / "trajectory_xt.npy", mmap_mode="r")
times = np.load(sample_dir / "trajectory_t.npy", mmap_mode="r")
final_state = np.load(sample_dir / "final_state.npy", mmap_mode="r")
if trajectory.ndim != 5:
    raise ValueError(f"{sample_dir}: trajectory_xt.npy shape should be (K,B,H,W,C), got {trajectory.shape}")
if final_state.ndim != 4:
    raise ValueError(f"{sample_dir}: final_state.npy shape should be (B,H,W,C), got {final_state.shape}")
if int(trajectory.shape[0]) != int(times.shape[0]):
    raise ValueError(f"{sample_dir}: trajectory length {trajectory.shape[0]} != trajectory_t length {times.shape[0]}")
if int(trajectory.shape[1]) != int(final_state.shape[0]):
    raise ValueError(f"{sample_dir}: trajectory batch {trajectory.shape[1]} != final_state batch {final_state.shape[0]}")
print(f"Validated sample: {sample_dir} | trajectory={tuple(trajectory.shape)} | final={tuple(final_state.shape)}")
PY
}

wait_batch() {
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || { echo "At least one attribution task failed; see ${LOG_ROOT}" >&2; exit 1; }
}

launch_attribution() {
  local task_id="$1"
  local query="$2"
  local algorithm="$3"
  local range="${4:-}"
  local slot="$5"
  local tag suffix output_dir
  tag="$(query_tag "${query}")"
  suffix="${algorithm}"
  output_dir="$(attribution_root_for_query "${query}")/${algorithm}_range_1_10000"
  if [[ "${algorithm}" == "traj_tracin" ]]; then
    suffix="${suffix}_${range//-/_}"
    output_dir="$(attribution_root_for_query "${query}")/$(traj_algorithm_tag)_range_${range//-/_}"
  fi
  if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${output_dir}" ]]; then
    echo "Refusing to overwrite ${output_dir}" >&2
    exit 1
  fi
  echo "Task ${task_id}: query=${query}, algorithm=${algorithm}, range=${range:-all}, slot=${slot}"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES="$((slot % GPU_PER_NODE))" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      INITIAL_SEED="${INITIAL_SEED}" \
      TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE}" \
      ALGORITHMS="${algorithm}" \
      ATTRIBUTION_RANGES="${range}" \
    bash scripts/01_data_attribution.sh >"${LOG_ROOT}/attr_task_${task_id}_${tag}_${suffix}.log" 2>&1 &
}

echo "Validating existing attribution samples"
for query in "${QUERIES[@]}"; do
  validate_sample "${query}"
done

echo "Attribution-only task range ${ATTR_TASK_START}-${ATTR_TASK_END}"
echo "TRAJ_QUERY_OBJECTIVE=${TRAJ_QUERY_OBJECTIVE}; ATTR_ALGORITHMS=${ATTR_ALGORITHMS}"
echo "MAX_PARALLEL_ATTR_TASKS=${MAX_PARALLEL_ATTR_TASKS}; GPU_PER_NODE=${GPU_PER_NODE}"
echo "Logs: ${LOG_ROOT}"

pids=()
task_id=0
launched=0
for query in "${QUERIES[@]}"; do
  for algorithm in "${ATTR_ALGORITHM_LIST[@]}"; do
    if [[ "${algorithm}" == "traj_tracin" ]]; then
      for range in "${TRAJ_RANGES[@]}"; do
        task_id=$((task_id + 1))
        if (( task_id < ATTR_TASK_START || task_id > ATTR_TASK_END )); then
          continue
        fi
        slot="${#pids[@]}"
        launch_attribution "${task_id}" "${query}" "${algorithm}" "${range}" "${slot}"
        pids+=("$!")
        launched=$((launched + 1))
        if (( ${#pids[@]} >= MAX_PARALLEL_ATTR_TASKS )); then
          wait_batch "${pids[@]}"
          pids=()
        fi
      done
    else
      task_id=$((task_id + 1))
      if (( task_id < ATTR_TASK_START || task_id > ATTR_TASK_END )); then
        continue
      fi
      slot="${#pids[@]}"
      launch_attribution "${task_id}" "${query}" "${algorithm}" "" "${slot}"
      pids+=("$!")
      launched=$((launched + 1))
      if (( ${#pids[@]} >= MAX_PARALLEL_ATTR_TASKS )); then
        wait_batch "${pids[@]}"
        pids=()
      fi
    fi
  done
done

if (( ${#pids[@]} > 0 )); then
  wait_batch "${pids[@]}"
fi

echo "Completed ${launched} attribution task(s) from range ${ATTR_TASK_START}-${ATTR_TASK_END}."
echo "Logs: ${LOG_ROOT}"
