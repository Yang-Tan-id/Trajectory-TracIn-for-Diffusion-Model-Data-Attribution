#!/usr/bin/env bash
#SBATCH -J x3-proj-query48-rtx
#SBATCH -o x3-proj-query48-rtx-%j.out
#SBATCH -e x3-proj-query48-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
X3_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REFINE_ROOT="$(cd "${X3_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${REFINE_ROOT}/.." && pwd)"

source "${X3_ROOT}/stampede3_das/_stampede3_das_lib.sh"
stampede3_das_init

PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
QUERY_PLAN_SEED="${QUERY_PLAN_SEED:-67}"
PROMPTED_INITIAL_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 11)}"
UNPROMPTED_INITIAL_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 11)}"
QUERY_PLAN_DIR="${X3_ROOT}/result/${EXPERIMENT_TAG}/query_plan"
QUERY_PLAN="${QUERY_PLAN:-${QUERY_PLAN_DIR}/query48.tsv}"
LOG_ROOT="${X3_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/projected_query48/${SLURM_JOB_ID:-local}"
mkdir -p "${QUERY_PLAN_DIR}" "${LOG_ROOT}"

export PROJECTED_CACHE_DIM

if [[ ! -f "${QUERY_PLAN}" ]]; then
  QUERY_PLAN="${QUERY_PLAN}" QUERY_PLAN_SEED="${QUERY_PLAN_SEED}" \
  PROMPTED_INITIAL_SEEDS="${PROMPTED_INITIAL_SEEDS_TEXT}" UNPROMPTED_INITIAL_SEEDS="${UNPROMPTED_INITIAL_SEEDS_TEXT}" \
  python3 - <<'PY'
import csv
import os
from pathlib import Path
import numpy as np

from dataset_config import CSV_PATH, LABEL_START

out = Path(os.environ["QUERY_PLAN"])
rng = np.random.default_rng(int(os.environ.get("QUERY_PLAN_SEED", "67")))
prompted_inits = [int(x) for x in os.environ["PROMPTED_INITIAL_SEEDS"].split()]
unprompted_inits = [int(x) for x in os.environ["UNPROMPTED_INITIAL_SEEDS"].split()]

rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.reader(f):
        if row and row[0].lower() != "id":
            labels = [x for x in row[LABEL_START:] if x]
            if labels:
                rows.append((row[0], labels))

if len(rows) < 36:
    raise SystemExit(f"Need at least 36 labeled rows, found {len(rows)}")

picked = rng.choice(len(rows), size=36, replace=False)
records = []
for seed in unprompted_inits:
    records.append(["unprompted_solo", "unprompted_solo", "unconditional", seed, 0, 1, "", "unconditional"])

j = 0
for seed in prompted_inits:
    for _ in range(3):
        row_id, labels = rows[int(picked[j])]
        query = ",".join(labels)
        sample_mode = "prompted_multi" if len(labels) > 1 else "prompted_solo"
        records.append([sample_mode, "prompted_solo", query, seed, 0, 0, row_id, query])
        j += 1

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["sample_mode", "score_mode", "query", "initial_seed", "sample_index", "unprompted", "source_row_id", "labels"])
    w.writerows(records)
print(f"[plan] wrote {len(records)} tasks -> {out}")
PY
fi

SWEEP="${X3_ROOT}/tacc/h100/projected_traj_tracin_score_sweep.sh"
pids=()
slot=0
while IFS=$'\t' read -r sample_mode score_mode query init sample_index unprompted source_row_id labels; do
  gpu=$((slot % 2))
  log="${LOG_ROOT}/task_${slot}__${score_mode}__seed_${init}__$(path_tag "${query}").log"
  echo "[task] slot=${slot} gpu=${gpu} sample_mode=${sample_mode} score_mode=${score_mode} query=${query} init=${init} row=${source_row_id} labels=${labels} -> ${log}"
  (
    for target_and_base in \
      "trajectory_next_checkpoint_noise_mse|${X3_ROOT}/result/${EXPERIMENT_TAG}/projected_traj_tracin_artifacts_next_ckpt" \
      "trajectory_next_checkpoint_ref_projection|${X3_ROOT}/result/${EXPERIMENT_TAG}/projected_traj_tracin_artifacts_refproj"; do
      IFS='|' read -r target artifact_base <<<"${target_and_base}"
      echo "[query] target=${target} artifact_base=${artifact_base}"
      env CUDA_VISIBLE_DEVICES="${gpu}" GPU_IDS="${gpu}" JAX_NUM_DEVICES=1 PYTHONUNBUFFERED=1 \
        PROJECTED_ARTIFACT_BASE="${artifact_base}" PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM}" PROJECTED_DIMS=4096 \
        RUN_TRAIN_STAGE=0 RUN_QUERY_STAGE=1 RUN_SCORE_SWEEP=0 SHARE_QUERY_ARTIFACT=1 SKIP_EXISTING_SCORES=1 \
        TRAJ_QUERY_OBJECTIVE="${target}" TRAJ_USE_SAVED_TRAJECTORY=1 \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
        SAMPLE_MODEL_MODE="${sample_mode}" ATTRIBUTION_SAMPLE_MODEL_MODE="${sample_mode}" ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}" \
        QUERY="${query}" INITIAL_SEED="${init}" SAMPLE_SEED="${init}" ATTRIBUTION_SAMPLE_INDEX="${sample_index}" UNPROMPTED="${unprompted}" \
        bash "${SWEEP}"
    done
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
  if (( ${#pids[@]} >= 2 )); then
    wait_all "${pids[@]}"
    pids=()
  fi
done < <(tail -n +2 "${QUERY_PLAN}")

if (( ${#pids[@]} > 0 )); then
  wait_all "${pids[@]}"
fi
echo "[done] x3 query48 projected artifacts"
