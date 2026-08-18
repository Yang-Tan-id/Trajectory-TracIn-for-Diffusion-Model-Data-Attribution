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

damping_tag_inner() {
  local value="$1"
  value="${value//+/_}"
  value="${value//-/neg_}"
  value="${value//./p}"
  tag_value "${value}"
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

score_dir_for_das_lambda() {
  local lambda="$1"
  local tag
  local pattern
  tag="$(damping_tag_inner "${lambda}")"
  if [[ "${UNPROMPTED}" == "1" ]]; then
    pattern=$(printf "%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/das_unprompted*/lambda_%s" \
      "${X3_ROOT}" "${EXPERIMENT_TAG}" "${SAMPLE_MODEL_MODE}" "${TRAIN_SEED}" "${INITIAL_SEED}" "${tag}"
    )
  else
    pattern=$(printf "%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/das*/lambda_%s" \
      "${X3_ROOT}" "${EXPERIMENT_TAG}" "${SAMPLE_MODEL_MODE}" "${TRAIN_SEED}" "$(tag_value "${QUERY}")" "${INITIAL_SEED}" "${tag}"
    )
  fi
  mapfile -t matches < <(compgen -G "${pattern}" | sort)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one DAS score dir for pattern ${pattern}, found ${#matches[@]}" >&2
    printf "  %s\n" "${matches[@]}" >&2
    exit 1
  fi
  printf "%s" "${matches[0]}"
}

eval_summary_for_call() {
  local lambda_tag="$1"
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
  printf "%s/result/%s/eval/%s/%s/initial_seed_%s/%s/das_lambda_%s/%s/%s/%s/lds_summary.json" \
    "${X3_ROOT}" "${EXPERIMENT_TAG}" "${SAMPLE_MODEL_MODE}" "${query_dir}" "${INITIAL_SEED}" \
    "${eval_kind}" "${lambda_tag}" "${target}" "${PRED_TAG:-pred_kept_sign_m1}" "$(basename "${model_dir}")"
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
  for target in ${TARGETS_TEXT}; do
    for lds_seed in ${LDS_SEEDS_TEXT}; do
      model_root="${X3_ROOT}/result/${EXPERIMENT_TAG}/lds_model/${SAMPLE_MODEL_MODE}/train_seed_${TRAIN_SEED}"
      model_pattern="${model_root}/m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_${lds_seed}"
      mapfile -t model_matches < <(compgen -G "${model_pattern}" | sort)
      if [[ "${#model_matches[@]}" -ne 1 ]]; then
        echo "Expected exactly one LDS model dir for pattern ${model_pattern}, found ${#model_matches[@]}" >&2
        printf "  %s\n" "${model_matches[@]}" >&2
        exit 1
      fi
      model_dir="${model_matches[0]}"
      for lambda in ${DAS_DAMPING_SWEEP_VALUES}; do
        if (( eval_slot_shard_count > 1 )); then
          if (( eval_call_index % eval_slot_shard_count != eval_slot_shard_index )); then
            eval_call_index=$((eval_call_index + 1))
            continue
          fi
        fi
        lambda_tag="$(damping_tag_inner "${lambda}")"
        score_dir="$(score_dir_for_das_lambda "${lambda}")"
        eval_summary="$(eval_summary_for_call "${lambda_tag}" "${target}" "${model_dir}")"
        if [[ -f "${eval_summary}" ]]; then
          echo "[lds-eval-skip] shard=${eval_slot_shard_index}/${eval_slot_shard_count} call=${eval_call_index} idx=${idx} mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED} target=${target} lambda=${lambda} lds_seed=${lds_seed} existing=${eval_summary}"
          eval_call_index=$((eval_call_index + 1))
          continue
        fi
        echo "[lds-eval] shard=${eval_slot_shard_index}/${eval_slot_shard_count} call=${eval_call_index} idx=${idx} mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED} target=${target} lambda=${lambda} lds_seed=${lds_seed}"
        if [[ "${UNPROMPTED}" == "1" ]]; then
          ATTRIBUTION_RESULT_DIRS="${score_dir}" LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --unprompted --algorithm "das_lambda_${lambda_tag}" --lds-model-dirs "${model_dir}" --target-function "${target}"
        else
          ATTRIBUTION_RESULT_DIRS="${score_dir}" LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --algorithm "das_lambda_${lambda_tag}" --lds-model-dirs "${model_dir}" --target-function "${target}"
        fi
        eval_call_index=$((eval_call_index + 1))
      done
    done
  done
done
