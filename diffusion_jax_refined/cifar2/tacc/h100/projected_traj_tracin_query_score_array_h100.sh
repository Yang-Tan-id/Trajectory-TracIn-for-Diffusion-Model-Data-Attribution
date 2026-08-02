#!/usr/bin/env bash
#SBATCH -J cifar2-proj-traj-query
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH -o cifar2-proj-traj-query-%j.out
#SBATCH -e cifar2-proj-traj-query-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_SLOTS="${GPU_SLOTS:-16}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
QUERY_TASK_SET="${QUERY_TASK_SET:-all}"
PROMPTED_INITIAL_SEEDS="${PROMPTED_INITIAL_SEEDS:-0 1 2 3 4 5 6 7}"
UNPROMPTED_INITIAL_SEEDS="${UNPROMPTED_INITIAL_SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23}"
PROMPTED_QUERIES="${PROMPTED_QUERIES:-horse automobile horse,automobile}"

export RUN_TRAIN_STAGE=0 RUN_QUERY_STAGE="${RUN_QUERY_STAGE:-1}" RUN_SCORE_SWEEP="${RUN_SCORE_SWEEP:-1}"

task_lines() {
  local seed query
  if [[ "${QUERY_TASK_SET}" != "prompted" && "${QUERY_TASK_SET}" != "unprompted" && "${QUERY_TASK_SET}" != "all" ]]; then
    echo "Unknown QUERY_TASK_SET=${QUERY_TASK_SET}; expected prompted, unprompted, or all." >&2
    exit 2
  fi
  if [[ "${QUERY_TASK_SET}" == "prompted" || "${QUERY_TASK_SET}" == "all" ]]; then
    for seed in ${PROMPTED_INITIAL_SEEDS}; do
      for query in ${PROMPTED_QUERIES}; do
        if [[ "${query}" == *","* ]]; then
          printf 'prompted_solo|prompted_multi|%s|%s|0\n' "${query}" "${seed}"
        else
          printf 'prompted_solo|prompted_solo|%s|%s|0\n' "${query}" "${seed}"
        fi
      done
    done
  fi
  if [[ "${QUERY_TASK_SET}" == "unprompted" || "${QUERY_TASK_SET}" == "all" ]]; then
    for seed in ${UNPROMPTED_INITIAL_SEEDS}; do
      printf 'unprompted_solo|unprompted_solo|unconditional|%s|1\n' "${seed}"
    done
  fi
}

run_task() {
  local slot="$1"
  local spec="$2"
  local score_mode sample_mode query seed unprompted_flag gpu backend
  IFS='|' read -r score_mode sample_mode query seed unprompted_flag <<<"${spec}"
  gpu="$((slot % GPU_PER_NODE))"
  backend="${TACC_SLOT_BACKEND:-ibrun}"

  cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu}"
    ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}"
    SAMPLE_MODEL_MODE="${sample_mode}"
    QUERY="${query}"
    INITIAL_SEED="${seed}"
    SAMPLE_SEED="${seed}"
    UNPROMPTED="${unprompted_flag}"
    RUN_TRAIN_STAGE=0
    bash "${SCRIPT_DIR}/projected_traj_tracin_score_sweep.sh"
  )
  echo "[slot ${slot}] score_mode=${score_mode} sample_mode=${sample_mode} query=${query} seed=${seed} gpu=${gpu}"
  if [[ "${backend}" == "ibrun" && -n "${SLURM_JOB_ID:-}" ]] && command -v ibrun >/dev/null; then
    ibrun -n 1 -o "${slot}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

worker() {
  local slot="$1"
  local i
  for ((i = slot; i < ${#TASKS[@]}; i += GPU_SLOTS)); do
    run_task "${slot}" "${TASKS[$i]}"
  done
}

TASKS=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && TASKS+=("${line}")
done < <(task_lines)

echo "Projected Traj-TracIn query+score array"
echo "task_set=${QUERY_TASK_SET}; tasks=${#TASKS[@]}; slots=${GPU_SLOTS}; gpu_per_node=${GPU_PER_NODE}"
echo "RUN_QUERY_STAGE=${RUN_QUERY_STAGE}; RUN_SCORE_SWEEP=${RUN_SCORE_SWEEP}"

pids=()
for ((slot = 0; slot < GPU_SLOTS; slot++)); do
  worker "${slot}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed != 0 )); then
  echo "At least one query+score worker failed." >&2
  exit 1
fi

echo "Projected query+score array complete."
