#!/usr/bin/env bash
set -euo pipefail

tag_value() {
  local value="$1"
  value="${value//,/__}"
  value="${value//+/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
  while [[ "${value}" == *"__"* ]]; do value="${value//__/_}"; done
  value="${value#_}"
  value="${value%_}"
  printf "%s" "${value:-unprompted}"
}

query_specs_inner() {
  local seed
  for seed in ${UNPROMPTED_SEEDS_TEXT}; do
    printf "unprompted_solo|unprompted|unconditional|%s|1\n" "${seed}"
  done
  for seed in ${PROMPTED_SEEDS_TEXT}; do
    printf "prompted_solo|horse|horse|%s|0\n" "${seed}"
    printf "prompted_solo|automobile|automobile|%s|0\n" "${seed}"
    printf "prompted_solo|horse,automobile|horse,automobile|%s|0\n" "${seed}"
  done
}

eval_summary_for_call() {
  local algorithm="$1"
  local target="$2"
  local model_dir="$3"
  local eval_kind
  local query_dir
  eval_kind="lds"
  query_dir="query_$(tag_value "${QUERY}")"
  if [[ "${UNPROMPTED}" == "1" ]]; then
    eval_kind="lds_unprompted"
    query_dir="unprompted"
  fi
  printf "%s/result/%s/eval/%s/%s/initial_seed_%s/%s/%s/%s/%s/%s/lds_summary.json" \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${SAMPLE_MODEL_MODE}" "${query_dir}" "${INITIAL_SEED}" \
    "${eval_kind}" "${algorithm}" "${target}" "${PRED_TAG:-pred_kept_sign_m1}" "$(basename "${model_dir}")"
}

mapfile -t specs < <(query_specs_inner)
eval_call_index=0
eval_slot_shard_index="${EVAL_SLOT_SHARD_INDEX:-0}"
eval_slot_shard_count="${EVAL_SLOT_SHARD_COUNT:-1}"
for ((idx = SLOT_INDEX; idx < ${#specs[@]}; idx += 16)); do
  IFS="|" read -r score_mode _query query_env seed unprompted_flag <<<"${specs[$idx]}"
  export SAMPLE_MODEL_MODE="${score_mode}"
  export UNPROMPTED_SAMPLE_MODEL_MODE="${score_mode}"
  export ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}"
  export UNPROMPTED_SCORE_MODEL_MODE="${score_mode}"
  export QUERY="${query_env}"
  export INITIAL_SEED="${seed}"
  export UNPROMPTED="${unprompted_flag}"
  algorithm_text="${EVAL_ALGORITHMS_TEXT}"
  if [[ "${UNPROMPTED}" == "1" && -n "${UNPROMPTED_EVAL_ALGORITHMS_TEXT:-}" ]]; then
    algorithm_text="${UNPROMPTED_EVAL_ALGORITHMS_TEXT}"
  fi
  for target in ${TARGETS_TEXT}; do
    for algorithm in ${algorithm_text}; do
      for lds_seed in ${LDS_SEEDS_TEXT}; do
        model_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/${SAMPLE_MODEL_MODE}/train_seed_${TRAIN_SEED}"
        model_pattern="${model_root}/m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_${lds_seed}"
        mapfile -t model_matches < <(compgen -G "${model_pattern}" | sort)
        if [[ "${#model_matches[@]}" -ne 1 ]]; then
          echo "Expected exactly one LDS model dir for pattern ${model_pattern}, found ${#model_matches[@]}" >&2
          printf "  %s\n" "${model_matches[@]}" >&2
          exit 1
        fi
        model_dir="${model_matches[0]}"
        if (( eval_slot_shard_count > 1 )); then
          if (( eval_call_index % eval_slot_shard_count != eval_slot_shard_index )); then
            eval_call_index=$((eval_call_index + 1))
            continue
          fi
        fi
        eval_summary="$(eval_summary_for_call "${algorithm}" "${target}" "${model_dir}")"
        if [[ "${FORCE_LDS_EVAL:-0}" != "1" && -f "${eval_summary}" ]]; then
          echo "[lds-eval-skip] shard=${eval_slot_shard_index}/${eval_slot_shard_count} call=${eval_call_index} idx=${idx} mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED} target=${target} algorithm=${algorithm} lds_seed=${lds_seed} existing=${eval_summary}"
          eval_call_index=$((eval_call_index + 1))
          continue
        fi
        echo "[lds-eval] shard=${eval_slot_shard_index}/${eval_slot_shard_count} call=${eval_call_index} idx=${idx} mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED} target=${target} algorithm=${algorithm} lds_seed=${lds_seed}"
        if [[ "${UNPROMPTED}" == "1" ]]; then
          LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --unprompted --algorithm "${algorithm}" --lds-model-dirs "${model_dir}" --target-function "${target}"
        else
          LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --algorithm "${algorithm}" --lds-model-dirs "${model_dir}" --target-function "${target}"
        fi
        eval_call_index=$((eval_call_index + 1))
      done
    done
  done
done
