#!/usr/bin/env bash
#SBATCH -J 3d-dtrak-3x100
#SBATCH -o 3d-dtrak-3x100-%j.out
#SBATCH -e 3d-dtrak-3x100-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
marker="diffusion_jax_refined/3dshapes/data_attribution/dtrak/01_train_datapoint_gradient.py"
if [[ ! -f "$repo/$marker" ]]; then
  candidate="$PWD"
  while [[ "$candidate" != / && ! -f "$candidate/$marker" ]]; do
    candidate="$(dirname "$candidate")"
  done
  repo="$candidate"
fi
[[ -f "$repo/$marker" ]] || { echo "cannot locate repository; set REPO_ROOT" >&2; exit 1; }

shapes="$repo/diffusion_jax_refined/3dshapes"
default_python="/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python"
[[ -x "$default_python" ]] || default_python="$(command -v python)"
python_bin="${PYTHON_BIN:-$default_python}"
experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
gpu_ids_text="${GPU_IDS:-0,1}"
IFS=, read -r -a gpu_ids <<< "$gpu_ids_text"
(( ${#gpu_ids[@]} == 2 )) || { echo "this job expects exactly two GPUs" >&2; exit 1; }

run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"
log_root="$shapes/result/$experiment/logs/dtrak_three_objectives_100x1/$run_tag"
mkdir -p "$log_root"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EXPERIMENT_TAG="$experiment" TRAIN_SEED="$train_seed" JAX_EPOCHS=200
export DATAPOINT_MODEL_MODE=prompted_solo SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SAMPLE_MODEL_MODE=prompted_solo ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export SAMPLE_ROOT="$shapes/result/$experiment/sample_ddim_eta0_1000"
export DTRAK_PROJ_DIM="${DTRAK_PROJ_DIM:-4096}"
export DTRAK_DAMPING="${DTRAK_DAMPING:-1e-3}"
export DTRAK_TRAIN_EXPECTATION_SAMPLES=100 DTRAK_QUERY_EXPECTATION_SAMPLES=100
export DTRAK_EXPLICIT_TIMESTEP_GRID=1 DTRAK_BATCH_SIZE="${DTRAK_BATCH_SIZE:-2}"
export DTRAK_COUNT_SKETCH_MODE="${DTRAK_COUNT_SKETCH_MODE:-scatter}"

train_stage="$shapes/data_attribution/dtrak/01_train_datapoint_gradient.py"
query_stage="$shapes/data_attribution/dtrak/02_query_gradient.py"
model_root="$shapes/result/$experiment/model/prompted_solo/seed_${train_seed}_train_gradient"

run_train_objective() {
  local objective="$1" gpu="$2"
  local artifact="$model_root/dtrak_${objective}_100x1/train_datapoint_gradient_artifact.npz"
  if [[ -f "$artifact" ]]; then
    echo "[skip] train $objective: $artifact"
    return
  fi
  echo "[train] objective=$objective gpu=$gpu artifact=$artifact"
  CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    DTRAK_OUTPUT_FUNCTION="$objective" \
    TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
    "$python_bin" "$train_stage"
}

# Two concurrent workers; GPU 0 takes two objectives sequentially and GPU 1 takes one.
(
  run_train_objective simple_loss "${gpu_ids[0]}"
  run_train_objective average "${gpu_ids[0]}"
) >"$log_root/train_gpu_${gpu_ids[0]}.log" 2>&1 &
train_pid0=$!
(
  run_train_objective square "${gpu_ids[1]}"
) >"$log_root/train_gpu_${gpu_ids[1]}.log" 2>&1 &
train_pid1=$!
wait "$train_pid0"
wait "$train_pid1"

mapfile -t selected_queries < <(
  "$python_bin" - "$query_file" "$query_ids" "$shapes" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[3])
from dataset_config import _prompt_tag
records = json.load(open(sys.argv[1]))["queries"]
for token in sys.argv[2].replace(",", " ").split():
    qid = int(token); row = records[qid]
    print(f"{qid}\t{row['prompt']}\t{int(row.get('initial_seed', row.get('seed')))}\t{_prompt_tag(row['prompt'])}")
PY
)

run_query_shard() {
  local shard="$1" gpu="$2" line qid prompt seed prompt_tag model_dir artifact objective
  for ((position=shard; position<${#selected_queries[@]}; position+=2)); do
    line="${selected_queries[$position]}"
    IFS=$'\t' read -r qid prompt seed prompt_tag <<< "$line"
    model_dir="$SAMPLE_ROOT/cifar/prompt_${prompt_tag}/model_prompted_solo__ckpt_seed_${train_seed}_epoch_0200"
    [[ -d "$model_dir/seed_$(printf '%06d' "$seed")" ]] || {
      echo "missing endpoint sample for Q$qid: $model_dir/seed_$(printf '%06d' "$seed")" >&2
      return 1
    }
    for objective in simple_loss square average; do
      artifact="$model_dir/seed_$(printf '%06d' "$seed")_query_gradient_dtrak_${objective}_100x1/dtrak/query_gradient_artifact.npz"
      if [[ -f "$artifact" ]]; then
        echo "[skip] query Q$qid objective=$objective"
        continue
      fi
      echo "[query] Q$qid objective=$objective gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
        QUERY="$prompt" INITIAL_SEED="$seed" SAMPLE_SEED="$seed" \
        ATTRIBUTION_SAMPLE_DIR="$model_dir" DTRAK_OUTPUT_FUNCTION="$objective" \
        QUERY_GRADIENT_ARTIFACT_PATH="$artifact" \
        "$python_bin" "$query_stage"
    done
  done
}

(cd "$repo"; run_query_shard 0 "${gpu_ids[0]}") >"$log_root/query_gpu_${gpu_ids[0]}.log" 2>&1 &
query_pid0=$!
(cd "$repo"; run_query_shard 1 "${gpu_ids[1]}") >"$log_root/query_gpu_${gpu_ids[1]}.log" 2>&1 &
query_pid1=$!
wait "$query_pid0"
wait "$query_pid1"

cd "$repo"
JAX_PLATFORMS=cpu "$python_bin" "$shapes/script/score_dtrak_three_objectives_100x1.py" \
  --experiment "$experiment" --train-seed "$train_seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  >"$log_root/score.log" 2>&1

if [[ "${RUN_LDS:-1}" == "1" ]]; then
  schemes="dtrak_simple_loss_train100x1_query100x1_q0_99 dtrak_square_train100x1_query100x1_q0_99 dtrak_average_train100x1_query100x1_q0_99"
  JAX_PLATFORMS=cpu "$python_bin" "$shapes/script/run_traj_tracin_lds_cached.py" \
    --execute --experiment "$experiment" --train-seed "$train_seed" \
    --query-file "$query_file" --query-ids "$query_ids" \
    --score-schemes "$schemes" --prediction-sign 1 \
    >"$log_root/lds.log" 2>&1
fi

echo "[done] D-TRAK simple_loss/square/average; train=query=100 timestamps x MC1; Q0-Q99"
