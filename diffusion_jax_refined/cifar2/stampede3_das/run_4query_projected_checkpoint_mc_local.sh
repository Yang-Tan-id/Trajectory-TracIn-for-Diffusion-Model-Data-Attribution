#!/usr/bin/env bash
set -euo pipefail

# Local/school-server runner: projected checkpoint-level train-loss variant.
# It reuses the existing projected raw artifacts directory by default and writes
# scores/eval under a separate algorithm folder. It does not create full-dim
# train/query gradient dumps.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CIFAR2_ROOT="${CIFAR2_ROOT:-${REPO_ROOT}/diffusion_jax_refined/cifar2}"
REFINE_ROOT="${REFINE_ROOT:-${REPO_ROOT}/diffusion_jax_refined}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment_67}"
TRAIN_SEED="${TRAIN_SEED:-67}"
GPU_SLOTS="${GPU_SLOTS:-4}"
RANGES="${RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"
QUERY_SPECS="${QUERY_SPECS:-unprompted_solo|unconditional|19|1 unprompted_solo|unconditional|5|1 prompted_solo|horse|5|0 prompted_solo|automobile|5|0}"
PROJECTED_ARTIFACT_DIR_NAME="${PROJECTED_ARTIFACT_DIR_NAME:-projected_traj_tracin_artifacts_raw}"
PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
TRAJ_TRACIN_TERM_WEIGHTING="${TRAJ_TRACIN_TERM_WEIGHTING:-uniform_checkpoint}"
SCORE_ALGORITHM_PREFIX="${SCORE_ALGORITHM_PREFIX:-traj_tracin_projected_checkpoint_mc}"
LDS_SEEDS="${LDS_SEEDS:-0 1 2 3 4 5 6 7}"
LDS_TARGETS="${LDS_TARGETS:-noise_trajectory simple_loss trajectory_state_mse projected_trajectory}"
PROJECTED_VARIANTS="${PROJECTED_VARIANTS:-raw query_l2_normalized train_l2_normalized query_train_l2_normalized}"

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

score_dir_for() {
  local mode="$1" query="$2" seed="$3" unprompted="$4" range="$5" variant="$6"
  local alg="${SCORE_ALGORITHM_PREFIX}_$(rtag "${range}")"
  if [[ "${unprompted}" == "1" ]]; then
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/%s/proj_%s/%s' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${mode}" "${TRAIN_SEED}" "${seed}" "${alg}" "${PROJECTED_DIMS}" "${variant}"
  else
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s/proj_%s/%s' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${mode}" "${TRAIN_SEED}" "$(qtag "${query}")" "${seed}" "${alg}" "${PROJECTED_DIMS}" "${variant}"
  fi
}

run_one() {
  local task_id="$1" gpu="$2" mode="$3" query="$4" seed="$5" unprompted="$6" range="$7"
  local query_env="${query}" alg log_root log
  [[ "${unprompted}" == "1" ]] && query_env="unconditional"
  alg="${SCORE_ALGORITHM_PREFIX}_$(rtag "${range}")"
  if [[ -f "$(score_dir_for "${mode}" "${query}" "${seed}" "${unprompted}" "${range}" raw)/scores.npy" ]]; then
    echo "[skip-score] ${mode} ${query} seed=${seed} ${range}"
    return 0
  fi
  log_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/local_logs/projected_checkpoint_mc_4query"
  mkdir -p "${log_root}"
  log="${log_root}/task_${task_id}__${mode}__$(qtag "${query}")__seed_${seed}__$(rtag "${range}").log"
  echo "[run-score] gpu=${gpu} mode=${mode} query=${query} seed=${seed} range=${range} -> ${log}"
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
    PROJECTED_ARTIFACT_DIR_NAME="${PROJECTED_ARTIFACT_DIR_NAME}" \
    PROJECTED_DIMS="${PROJECTED_DIMS}" PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM}" \
    TRAIN_SCORE_INDEX_RANGES="${range}" PROJECTED_SCORE_INDEX_RANGES="${range}" \
    TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse \
    TRAJ_PARAMETER_SOURCE=raw \
    TRAJ_TRACIN_TERM_WEIGHTING="${TRAJ_TRACIN_TERM_WEIGHTING}" \
    RUN_TRAIN_STAGE=0 RUN_QUERY_STAGE=0 RUN_SCORE_SWEEP=1 SKIP_EXISTING_SCORES=0 \
    SCORE_ALGORITHM_DIR="${alg}" \
    QUERY="${query_env}" INITIAL_SEED="${seed}" SAMPLE_SEED="${seed}" SAMPLE_SEEDS="${seed}" \
    SAMPLE_MODEL_MODE="${mode}" ATTRIBUTION_SAMPLE_MODEL_MODE="${mode}" UNPROMPTED_SAMPLE_MODEL_MODE="${mode}" \
    ATTRIBUTION_SCORE_MODEL_MODE="${mode}" UNPROMPTED_SCORE_MODEL_MODE="${mode}" \
    UNPROMPTED="${unprompted}" \
    PYTHONUNBUFFERED=1 bash "${CIFAR2_ROOT}/tacc/h100/projected_traj_tracin_score_sweep.sh"
  ) >"${log}" 2>&1
}

variant_algorithm() {
  local variant="$1"
  case "${variant}" in
    raw) printf '%s_raw' "${SCORE_ALGORITHM_PREFIX}" ;;
    query_l2_normalized) printf '%s_q_l2' "${SCORE_ALGORITHM_PREFIX}" ;;
    train_l2_normalized) printf '%s_t_l2' "${SCORE_ALGORITHM_PREFIX}" ;;
    query_train_l2_normalized) printf '%s_qt_l2' "${SCORE_ALGORITHM_PREFIX}" ;;
    *) printf '%s_%s' "${SCORE_ALGORITHM_PREFIX}" "$(qtag "${variant}")" ;;
  esac
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
  local mode="$1" query="$2" seed="$3" unprompted="$4" variant="$5"
  local alg score_dirs mode_arg eval_root target lds_seed target_csv out_dir
  alg="$(variant_algorithm "${variant}")"
  score_dirs=""
  for range in ${RANGES}; do
    local dir
    dir="$(score_dir_for "${mode}" "${query}" "${seed}" "${unprompted}" "${range}" "${variant}")"
    [[ -f "${dir}/scores.npy" ]] || return 0
    score_dirs="${score_dirs:+${score_dirs},}${dir}"
  done
  if [[ "${unprompted}" == "1" ]]; then
    mode_arg="unprompted"
    eval_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/unprompted/initial_seed_${seed}/lds_unprompted/${alg}"
  else
    mode_arg="prompted"
    eval_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/query_$(qtag "${query}")/initial_seed_${seed}/lds/${alg}"
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
        --algorithm "${alg}" \
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
  local mode="$1" query="$2" seed="$3" unprompted="$4" variant="$5"
  local alg root
  alg="$(variant_algorithm "${variant}")"
  if [[ "${unprompted}" == "1" ]]; then
    root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/unprompted/initial_seed_${seed}/lds_unprompted/${alg}"
  else
    root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}/query_$(qtag "${query}")/initial_seed_${seed}/lds/${alg}"
  fi
  "${PYTHON_BIN}" - "${mode}" "${query}" "${seed}" "${alg}" "${root}" <<'PY'
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

echo "Projected checkpoint-MC 4-query score run"
echo "queries=${QUERY_SPECS}"
echo "ranges=${RANGES}; gpus=${GPU_SLOTS}; artifact_dir=${PROJECTED_ARTIFACT_DIR_NAME}; term_weighting=${TRAJ_TRACIN_TERM_WEIGHTING}"

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
  for variant in ${PROJECTED_VARIANTS}; do
    eval_one_group "${mode}" "${query}" "${seed}" "${unprompted}" "${variant}"
    summarize_group "${mode}" "${query}" "${seed}" "${unprompted}" "${variant}"
  done
done

echo "Done."
