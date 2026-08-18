#!/usr/bin/env bash
set -euo pipefail

# One-command X3 local/server workflow:
#   1) sample the three X3 query prompts with the requested initial seed;
#   2) run traj_tracin with the L1 eps-deviation objective for all query/range shards;
#   3) evaluate those scores against existing k=8000 LDS models, separately by LDS seed;
#   4) aggregate per-seed LDS summaries and scatter SVGs.
#
# This script is intended for a regular GPU server, not Slurm/TACC. It creates
# or reuses query samples and reuses existing LDS models.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-eps_deviation_l1_mean}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-8000}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-noise_trajectory}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_ATTRIBUTION="${RUN_ATTRIBUTION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
SAMPLE_TRAJECTORY_STEPS="${SAMPLE_TRAJECTORY_STEPS:-100}"

QUERIES_TEXT="${QUERIES_TEXT:-horse automobile horse,automobile}"
TRAJ_RANGES_TEXT="${TRAJ_RANGES_TEXT:-1-2500 2501-5000 5001-7500 7501-10000}"
read -r -a QUERIES <<<"${QUERIES_TEXT}"
read -r -a TRAJ_RANGES <<<"${TRAJ_RANGES_TEXT}"

RAW_CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
read -r -a GPU_IDS <<<"${RAW_CUDA_DEVICES//,/ }"
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
MAX_PARALLEL_SAMPLE_TASKS="${MAX_PARALLEL_SAMPLE_TASKS:-${NUM_GPUS}}"
MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${NUM_GPUS}}"
MAX_PARALLEL_EVAL_TASKS="${MAX_PARALLEL_EVAL_TASKS:-${NUM_GPUS}}"

LOG_ROOT="${ROOT}/result/${EXPERIMENT_TAG}/local_logs/traj_l1_k${LDS_K}_m${LDS_M}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"

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

query_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//[^A-Za-z0-9._-]/_}"
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

prediction_tag() {
  local subset="$1"
  local sign="$2"
  local sign_text
  sign_text="$(python - "${sign}" <<'PY'
import sys
x = float(sys.argv[1])
print(str(int(x)) if x.is_integer() else f"{x:g}")
PY
)"
  sign_text="${sign_text//-/m}"
  sign_text="${sign_text//+/p}"
  sign_text="${sign_text//./p}"
  printf 'pred_%s_sign_%s' "${subset}" "${sign_text}"
}

gpu_for_slot() {
  local slot="$1"
  local idx=$((slot % NUM_GPUS))
  printf '%s' "${GPU_IDS[$idx]}"
}

wait_batch() {
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || {
    echo "At least one background task failed. Check ${LOG_ROOT}." >&2
    exit 1
  }
}

sample_dir_for_query() {
  local query="$1"
  printf '%s/result/%s/eval/sampling/cifar/prompt_%s/model_prompted_jax__ckpt_seed_42_epoch_0200/seed_%06d' \
    "${ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "${query}")" "${INITIAL_SEED}"
}

sample_complete() {
  local query="$1"
  local sample_dir
  sample_dir="$(sample_dir_for_query "${query}")"
  [[ -f "${sample_dir}/trajectory_xt.npy" ]] &&
    [[ -f "${sample_dir}/trajectory_t.npy" ]] &&
    [[ -f "${sample_dir}/final_state.npy" ]] &&
    [[ -f "${sample_dir}/seed_info.json" ]]
}

run_sample() {
  echo "Running/validating sampling for queries: ${QUERIES[*]}"
  local pids=()
  local query tag sample_dir slot
  for query in "${QUERIES[@]}"; do
    tag="$(query_tag "${query}")"
    sample_dir="$(sample_dir_for_query "${query}")"
    if [[ "${ALLOW_OVERWRITE}" != "1" ]] && sample_complete "${query}"; then
      echo "Skip existing sample: ${sample_dir}"
      continue
    fi
    if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${sample_dir}" ]]; then
      echo "Refusing to use incomplete existing sample folder: ${sample_dir}" >&2
      echo "Either remove it manually or rerun with ALLOW_OVERWRITE=1." >&2
      exit 1
    fi
    slot="${#pids[@]}"
    echo "Launch sample query=${query} seed=${INITIAL_SEED} gpu=$(gpu_for_slot "${slot}")"
    (
      cd "${ROOT}"
      CUDA_VISIBLE_DEVICES="$(gpu_for_slot "${slot}")" \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
        QUERY="${query}" \
        INITIAL_SEED="${INITIAL_SEED}" \
        SAMPLE_SEEDS="${INITIAL_SEED}" \
        SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE}" \
        SAMPLE_TRAJECTORY_STEPS="${SAMPLE_TRAJECTORY_STEPS}" \
        bash scripts/00_sample.sh
    ) >"${LOG_ROOT}/sample_${tag}.log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL_SAMPLE_TASKS )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    wait_batch "${pids[@]}"
  fi
}

validate_samples() {
  echo "Validating existing samples..."
  for query in "${QUERIES[@]}"; do
    local sample_dir
    sample_dir="$(sample_dir_for_query "${query}")"
    [[ -f "${sample_dir}/trajectory_xt.npy" ]] || { echo "Missing ${sample_dir}/trajectory_xt.npy" >&2; exit 1; }
    [[ -f "${sample_dir}/trajectory_t.npy" ]] || { echo "Missing ${sample_dir}/trajectory_t.npy" >&2; exit 1; }
    [[ -f "${sample_dir}/seed_info.json" ]] || { echo "Missing ${sample_dir}/seed_info.json" >&2; exit 1; }
  done
}

score_dir_for_range() {
  local query="$1"
  local range="$2"
  printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s_range_%s' \
    "${ROOT}" "${EXPERIMENT_TAG}" "${ATTRIBUTION_SCORE_MODEL_MODE:-${SAMPLE_MODEL_MODE:-prompted_solo}}" "${TRAIN_SEED:-42}" "$(path_tag "${query}")" "${INITIAL_SEED}" \
    "$(traj_algorithm_tag)" "${range//-/_}"
}

score_dirs_for_query() {
  local query="$1"
  local output=""
  local range
  for range in "${TRAJ_RANGES[@]}"; do
    output+="${output:+,}$(score_dir_for_range "${query}" "${range}")"
  done
  printf '%s' "${output}"
}

run_attribution() {
  echo "Running traj_tracin attribution objective=${TRAJ_QUERY_OBJECTIVE}"
  local pids=()
  local query range slot tag output_dir
  for query in "${QUERIES[@]}"; do
    tag="$(query_tag "${query}")"
    for range in "${TRAJ_RANGES[@]}"; do
      output_dir="$(score_dir_for_range "${query}" "${range}")"
      if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${output_dir}/scores.npy" ]]; then
        echo "Skip existing attribution: ${output_dir}"
        continue
      fi
      if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${output_dir}" && ! -f "${output_dir}/scores.npy" ]]; then
        echo "Refusing to use incomplete existing attribution folder: ${output_dir}" >&2
        exit 1
      fi
      slot="${#pids[@]}"
      echo "Launch attribution query=${query} range=${range} gpu=$(gpu_for_slot "${slot}")"
      (
        cd "${ROOT}"
        CUDA_VISIBLE_DEVICES="$(gpu_for_slot "${slot}")" \
          EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
          QUERY="${query}" \
          INITIAL_SEED="${INITIAL_SEED}" \
          TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE}" \
          ALGORITHMS="traj_tracin" \
          ATTRIBUTION_RANGES="${range}" \
          bash scripts/01_data_attribution.sh
      ) >"${LOG_ROOT}/attr_${tag}_${range//-/_}.log" 2>&1 &
      pids+=("$!")
      if (( ${#pids[@]} >= MAX_PARALLEL_ATTR_TASKS )); then
        wait_batch "${pids[@]}"
        pids=()
      fi
    done
  done
  if (( ${#pids[@]} > 0 )); then
    wait_batch "${pids[@]}"
  fi
}

lds_model_dir_for_seed() {
  local seed="$1"
  printf '%s/result/%s/lds_model/m_%s_k_%s_seed_%s' \
    "${ROOT}" "${EXPERIMENT_TAG}" "${LDS_M}" "${LDS_K}" "${seed}"
}

validate_lds_models() {
  echo "Validating LDS models M=${LDS_M}, K=${LDS_K}, seeds=${LDS_SEEDS}"
  local seed model_dir
  for seed in ${LDS_SEEDS}; do
    model_dir="$(lds_model_dir_for_seed "${seed}")"
    [[ -f "${model_dir}/lds_model_config.json" ]] || { echo "Missing ${model_dir}/lds_model_config.json" >&2; exit 1; }
    grep -q '"complete": true' "${model_dir}/lds_model_config.json" || {
      echo "Incomplete LDS model folder: ${model_dir}" >&2
      exit 1
    }
  done
}

run_eval() {
  echo "Running per-seed LDS eval for $(traj_algorithm_tag)"
  local pids=()
  local pred_tag
  pred_tag="$(prediction_tag "${LDS_PREDICTION_SUBSET}" "${LDS_PREDICTION_SIGN}")"
  local query tag seed model_dir dirs out_dir slot
  for query in "${QUERIES[@]}"; do
    tag="$(path_tag "${query}")"
    dirs="$(score_dirs_for_query "${query}")"
    IFS=',' read -r -a inputs <<<"${dirs}"
    for input in "${inputs[@]}"; do
      [[ -f "${input}/scores.npy" ]] || { echo "Missing attribution scores: ${input}/scores.npy" >&2; exit 1; }
    done
    for seed in ${LDS_SEEDS}; do
      model_dir="$(lds_model_dir_for_seed "${seed}")"
      out_dir="${ROOT}/result/${EXPERIMENT_TAG}/eval/${ATTRIBUTION_SCORE_MODEL_MODE:-${SAMPLE_MODEL_MODE:-prompted_solo}}/query_${tag}/initial_seed_${INITIAL_SEED}/lds/$(traj_algorithm_tag)/${LDS_TARGET_FUNCTION}/${pred_tag}/$(basename "${model_dir}")"
      if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${out_dir}/lds_results.csv" ]]; then
        echo "Skip existing eval: ${out_dir}"
        continue
      fi
      slot="${#pids[@]}"
      echo "Launch LDS eval query=${query} seed=${seed} gpu=$(gpu_for_slot "${slot}")"
      (
        cd "${ROOT}"
        CUDA_VISIBLE_DEVICES="$(gpu_for_slot "${slot}")" \
          EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
          QUERY="${query}" \
          INITIAL_SEED="${INITIAL_SEED}" \
          ALGORITHMS="$(traj_algorithm_tag)" \
          ATTRIBUTION_RESULT_DIRS="${dirs}" \
          LDS_MODEL_DIRS="${model_dir}" \
          LDS_DEVICE=gpu \
          LDS_NUM_DEVICES=1 \
          bash scripts/04_lds_eval.sh \
            --target-function "${LDS_TARGET_FUNCTION}" \
            --prediction-subset "${LDS_PREDICTION_SUBSET}" \
            --prediction-sign "${LDS_PREDICTION_SIGN}"
      ) >"${LOG_ROOT}/eval_${tag}_seed_${seed}.log" 2>&1 &
      pids+=("$!")
      if (( ${#pids[@]} >= MAX_PARALLEL_EVAL_TASKS )); then
        wait_batch "${pids[@]}"
        pids=()
      fi
    done
  done
  if (( ${#pids[@]} > 0 )); then
    wait_batch "${pids[@]}"
  fi
}

run_aggregate() {
  echo "Aggregating per-seed LDS evals"
  local pred_tag
  pred_tag="$(prediction_tag "${LDS_PREDICTION_SUBSET}" "${LDS_PREDICTION_SIGN}")"
  python "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
    --eval-root "${ROOT}/result/${EXPERIMENT_TAG}/eval" \
    --target-function "${LDS_TARGET_FUNCTION}" \
    --lds-m "${LDS_M}" \
    --lds-k "${LDS_K}" \
    --initial-seed "${INITIAL_SEED}" \
    --algorithms "$(traj_algorithm_tag)" \
    --prediction-dir "${pred_tag}" \
    --output-name "aggregate_m_${LDS_M}_k_${LDS_K}_${pred_tag}_traj_l1_seeds_${LDS_SEEDS// /_}" \
    >"${LOG_ROOT}/aggregate.log" 2>&1
  echo "Aggregate log: ${LOG_ROOT}/aggregate.log"
}

echo "X3 traj L1 attribution + k=${LDS_K} LDS eval"
echo "ROOT=${ROOT}"
echo "EXPERIMENT_TAG=${EXPERIMENT_TAG}; INITIAL_SEED=${INITIAL_SEED}"
echo "TRAJ_QUERY_OBJECTIVE=${TRAJ_QUERY_OBJECTIVE}; algorithm folder=$(traj_algorithm_tag)"
echo "CUDA devices=${GPU_IDS[*]}; sample_parallel=${MAX_PARALLEL_SAMPLE_TASKS}; attr_parallel=${MAX_PARALLEL_ATTR_TASKS}; eval_parallel=${MAX_PARALLEL_EVAL_TASKS}"
echo "Logs: ${LOG_ROOT}"

if [[ "${RUN_SAMPLE}" == "1" ]]; then
  run_sample
else
  echo "RUN_SAMPLE=0, skipping sampling"
fi

validate_samples
if [[ "${RUN_ATTRIBUTION}" == "1" ]]; then
  run_attribution
else
  echo "RUN_ATTRIBUTION=0, skipping attribution"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  validate_lds_models
  run_eval
else
  echo "RUN_EVAL=0, skipping LDS eval"
fi

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  run_aggregate
else
  echo "RUN_AGGREGATE=0, skipping aggregate"
fi

echo "Done. Logs are in ${LOG_ROOT}"
