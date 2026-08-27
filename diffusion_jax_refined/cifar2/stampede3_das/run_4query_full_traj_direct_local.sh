#!/usr/bin/env bash
set -euo pipefail

# Local/school-server runner: full Traj-TracIn, no gradient artifacts.
# It scores four hand-picked queries on four GPUs, then fast-evaluates against
# existing cached LDS target CSVs.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CIFAR2_ROOT="${CIFAR2_ROOT:-${REPO_ROOT}/diffusion_jax_refined/cifar2}"
REFINE_ROOT="${REFINE_ROOT:-${REPO_ROOT}/diffusion_jax_refined}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment_67}"
TRAIN_SEED="${TRAIN_SEED:-67}"
GPU_SLOTS="${GPU_SLOTS:-4}"
RANGES="${RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"
LDS_SEEDS="${LDS_SEEDS:-0 1 2 3 4 5 6 7}"
LDS_TARGETS="${LDS_TARGETS:-noise_trajectory simple_loss trajectory_state_mse projected_trajectory}"
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_next_checkpoint_noise_mse}"
TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE:-raw}"
TRACIN_USE_LR_WEIGHTS="${TRACIN_USE_LR_WEIGHTS:-0}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-2}"
TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"
TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS="${TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS:-1}"
TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS="${TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS:-10}"

# mode|query|seed|unprompted_flag
QUERY_SPECS="${QUERY_SPECS:-unprompted_solo|unconditional|19|1 unprompted_solo|unconditional|5|1 prompted_solo|horse|5|0 prompted_solo|automobile|5|0}"

export PYTHONPATH="${REFINE_ROOT}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CUDNN_USE_AUTOTUNE="${TF_CUDNN_USE_AUTOTUNE:-0}"

qtag() {
  local text="$1"
  text="${text//,/__}"
  text="${text//[^A-Za-z0-9._-]/_}"
  while [[ "${text}" == *"__"* ]]; do text="${text//__/_}"; done
  text="${text#_}"
  text="${text%_}"
  printf '%s' "${text:-unprompted}"
}

rtag() {
  local range="$1"
  range="${range//:/-}"
  printf 'range_%s_%s' "${range%-*}" "${range#*-}"
}

alg_base_for() {
  local unprompted="$1"
  if [[ "${unprompted}" == "1" ]]; then
    printf 'traj_tracin_full_direct_avg100mc_nextckpt_raw_unprompted'
  else
    printf 'traj_tracin_full_direct_avg100mc_nextckpt_raw'
  fi
}

score_dir_for() {
  local mode="$1" query="$2" seed="$3" unprompted="$4" range="$5"
  local alg_base alg
  alg_base="$(alg_base_for "${unprompted}")"
  alg="${alg_base}_$(rtag "${range}")"
  if [[ "${unprompted}" == "1" ]]; then
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/%s' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${mode}" "${TRAIN_SEED}" "${seed}" "${alg}"
  else
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${mode}" "${TRAIN_SEED}" "$(qtag "${query}")" "${seed}" "${alg}"
  fi
}

ensure_sample() {
  local mode="$1" query="$2" seed="$3" unprompted="$4"
  local query_env="${query}"
  [[ "${unprompted}" == "1" ]] && query_env="unconditional"
  EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
  SAMPLE_MODEL_MODE="${mode}" ATTRIBUTION_SAMPLE_MODEL_MODE="${mode}" UNPROMPTED_SAMPLE_MODEL_MODE="${mode}" \
  QUERY="${query_env}" INITIAL_SEED="${seed}" SAMPLE_SEED="${seed}" SAMPLE_SEEDS="${seed}" \
    bash "${CIFAR2_ROOT}/scripts/00_sample_for_attribution.sh"
}

run_one() {
  local task_id="$1" gpu="$2" mode="$3" query="$4" seed="$5" unprompted="$6" range="$7"
  local out_dir alg query_env log_root log
  out_dir="$(score_dir_for "${mode}" "${query}" "${seed}" "${unprompted}" "${range}")"
  if [[ -f "${out_dir}/scores.npy" ]]; then
    echo "[skip-score] ${out_dir}/scores.npy"
    return 0
  fi
  alg="$(basename "${out_dir}")"
  query_env="${query}"
  [[ "${unprompted}" == "1" ]] && query_env="unconditional"
  log_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/local_logs/full_traj_direct_4query"
  mkdir -p "${log_root}"
  log="${log_root}/task_${task_id}__${mode}__$(qtag "${query}")__seed_${seed}__$(rtag "${range}").log"
  echo "[run-score] gpu=${gpu} mode=${mode} query=${query} seed=${seed} range=${range} -> ${log}"
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
    SAMPLE_MODEL_MODE="${mode}" ATTRIBUTION_SAMPLE_MODEL_MODE="${mode}" UNPROMPTED_SAMPLE_MODEL_MODE="${mode}" \
    ATTRIBUTION_SCORE_MODEL_MODE="${mode}" UNPROMPTED_SCORE_MODEL_MODE="${mode}" \
    QUERY="${query_env}" INITIAL_SEED="${seed}" SAMPLE_SEED="${seed}" SAMPLE_SEEDS="${seed}" \
    UNPROMPTED="${unprompted}" \
    SCORE_INDEX_RANGES="${range}" ATTRIBUTION_RANGES="${range}" \
    TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE}" TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE}" \
    TRACIN_USE_LR_WEIGHTS="${TRACIN_USE_LR_WEIGHTS}" \
    TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}" \
    TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE}" \
    TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS="${TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS}" \
    TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS="${TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS}" \
    TRAJ_SAVE_QUERY_NORMALIZED_SCORES=0 \
    TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS="" \
    ATTRIBUTION_OUTPUT_ALGORITHM="${alg}" \
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py" \
      "${CIFAR2_ROOT}/data_attribution/traj_tracin/CONFIG.py"
  ) >"${log}" 2>&1
}

find_target_csv() {
  local mode="$1" query="$2" seed="$3" unprompted="$4" target="$5" lds_seed="$6"
  local root
  if [[ "${unprompted}" == "1" ]]; then
    root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/unprompted/initial_seed_${seed}/lds_unprompted"
  else
    root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/query_$(qtag "${query}")/initial_seed_${seed}/lds"
  fi
  find "${root}" -path "*/${target}/pred_kept_sign_m1/m_64_k_2500_pct_25_subset_seed_${lds_seed}/lds_results.csv" 2>/dev/null | head -1
}

eval_one_group() {
  local mode="$1" query="$2" seed="$3" unprompted="$4"
  local alg_base score_dirs mode_arg eval_root target lds_seed target_csv out_dir summary
  alg_base="$(alg_base_for "${unprompted}")"
  score_dirs=""
  for range in ${RANGES}; do
    local dir
    dir="$(score_dir_for "${mode}" "${query}" "${seed}" "${unprompted}" "${range}")"
    [[ -f "${dir}/scores.npy" ]] || return 0
    score_dirs="${score_dirs:+${score_dirs},}${dir}"
  done
  if [[ "${unprompted}" == "1" ]]; then
    mode_arg="unprompted"
    eval_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/unprompted/initial_seed_${seed}/lds_unprompted/${alg_base}"
  else
    mode_arg="prompted"
    eval_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/query_$(qtag "${query}")/initial_seed_${seed}/lds/${alg_base}"
  fi
  for target in ${LDS_TARGETS}; do
    for lds_seed in ${LDS_SEEDS}; do
      target_csv="$(find_target_csv "${mode}" "${query}" "${seed}" "${unprompted}" "${target}" "${lds_seed}")"
      [[ -n "${target_csv}" ]] || continue
      out_dir="${eval_root}/${target}/pred_kept_sign_m1/m_64_k_2500_pct_25_subset_seed_${lds_seed}"
      "${PYTHON_BIN}" "${REFINE_ROOT}/common/fast_lds_score_eval.py" \
        "${CIFAR2_ROOT}/dataset_config.py" \
        --target-results "${target_csv}" \
        --score-file "${score_dirs}" \
        --algorithm "${alg_base}" \
        --out-dir "${out_dir}" \
        --prediction-subset kept \
        --prediction-sign -1 \
        --duplicate-policy max \
        --target-function "${target}" \
        --mode "${mode_arg}" >/dev/null
    done
  done
}

summarize_group() {
  local mode="$1" query="$2" seed="$3" unprompted="$4"
  local alg_base root
  alg_base="$(alg_base_for "${unprompted}")"
  if [[ "${unprompted}" == "1" ]]; then
    root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/unprompted/initial_seed_${seed}/lds_unprompted/${alg_base}"
  else
    root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/query_$(qtag "${query}")/initial_seed_${seed}/lds/${alg_base}"
  fi
  "${PYTHON_BIN}" - "${mode}" "${query}" "${seed}" "${alg_base}" "${root}" <<'PY'
import json, sys
from pathlib import Path
mode, query, seed, alg, root = sys.argv[1:]
rows = []
for p in Path(root).glob("*/pred_kept_sign_m1/m_64_k_2500_pct_25_subset_seed_*/lds_summary.json"):
    d = json.loads(p.read_text())
    rows.append((d.get("target_function", p.parts[-4]), float(d["lds_spearman"])))
by = {}
for target, value in rows:
    by.setdefault(target, []).append(value)
for target, vals in sorted(by.items()):
    print(f"{alg:42s} {mode:16s} {query:16s} seed={seed:>2s} {target:24s} n={len(vals):2d} mean={sum(vals)/len(vals):+.4f}")
PY
}

echo "Full direct Traj-TracIn 4-query run"
echo "queries=${QUERY_SPECS}"
echo "ranges=${RANGES}; gpus=${GPU_SLOTS}; lr_weights=${TRACIN_USE_LR_WEIGHTS}; objective=${TRAJ_QUERY_OBJECTIVE}"
echo "aggregate_train_timestamps=${TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS}; aggregate_num_timesteps=${TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS}; mc_per_timestep=10"

tasks=()
for spec in ${QUERY_SPECS}; do
  IFS='|' read -r mode query seed unprompted <<<"${spec}"
  for range in ${RANGES}; do
    tasks+=("${mode}|${query}|${seed}|${unprompted}|${range}")
  done
done

pids=()
for i in "${!tasks[@]}"; do
  IFS='|' read -r mode query seed unprompted range <<<"${tasks[$i]}"
  gpu=$(( i % GPU_SLOTS ))
  run_one "${i}" "${gpu}" "${mode}" "${query}" "${seed}" "${unprompted}" "${range}" &
  pids+=("$!")
  if (( ${#pids[@]} >= GPU_SLOTS )); then
    for pid in "${pids[@]}"; do wait "${pid}"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "${pid}"; done

for spec in ${QUERY_SPECS}; do
  IFS='|' read -r mode query seed unprompted <<<"${spec}"
  eval_one_group "${mode}" "${query}" "${seed}" "${unprompted}"
  summarize_group "${mode}" "${query}" "${seed}" "${unprompted}"
done

echo "Done."
