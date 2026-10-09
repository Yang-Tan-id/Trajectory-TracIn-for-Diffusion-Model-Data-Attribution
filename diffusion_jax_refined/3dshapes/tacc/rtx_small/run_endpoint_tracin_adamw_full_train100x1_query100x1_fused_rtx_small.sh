#!/usr/bin/env bash
#SBATCH -J 3d-end-a100-q100
#SBATCH -o 3d-end-a100-q100-%j.out
#SBATCH -e 3d-end-a100-q100-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
marker="diffusion_jax_refined/3dshapes/script/generate_endpoint_simple_loss_query100x1_bank.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do repo="$(dirname "$repo")"; done
[[ -f "$repo/$marker" ]] || {
  echo "cannot locate repository; set REPO_ROOT" >&2
  exit 1
}

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
(( ${#gpu_ids[@]} == 2 )) || { echo "this RTX job expects exactly two GPUs" >&2; exit 1; }

# The current Slurm id must override any stale exported RUN_TAG.
run_tag="${SLURM_JOB_ID:-${RUN_TAG:-manual_$(date +%Y%m%d_%H%M%S)}}"
log_root="$shapes/result/$experiment/logs/endpoint_tracin_adamw_full_train100x1_query100x1_fused/$run_tag"
# A retry can explicitly reuse checkpoint-level score components from an
# earlier job without mixing the two jobs' logs: export RESUME_TAG=<old_job_id>.
component_tag="${RESUME_TAG:-$run_tag}"
component_root="$shapes/result/$experiment/logs/endpoint_tracin_adamw_full_train100x1_query100x1_fused/$component_tag"
component_prefix="$component_root/components"
query_artifact_list="$log_root/query_artifacts.txt"
query_namespace="endpoint_tracin_simple_loss_mean100t_mc1_q0_99"
score_namespace="endpoint_tracin_adamw_full_train100x1_query100x1_q0_99"
mkdir -p "$log_root" "$component_root"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# RTX CUDA workers must be real Slurm tasks; otherwise TACC cancels the batch
# shell at roughly 00:02:02 with `CANCELLED by 0`.
run_task() {
  local task_offset="$1" gpu="$2"
  shift 2
  ibrun -n 1 -o "$task_offset" env \
    CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 "$@"
}

if [[ "${PREPARE_SAMPLES:-1}" == "1" ]]; then
  echo "[sample] verify/generate Q0-Q99 endpoints"
  ibrun -n 1 -o 0 env \
    CUDA_VISIBLE_DEVICES="$gpu_ids_text" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    "$python_bin" "$shapes/script/run_traj_tracin_queries_and_scores.py" \
    --execute --experiment "$experiment" --train-seed "$train_seed" --epochs 200 \
    --query-file "$query_file" --query-ids "$query_ids" \
    --gpus "$gpu_ids_text" --python-bin "$python_bin" \
    --skip-query-gradient --skip-score \
    >"$log_root/sample.log" 2>&1
fi

query_bank="$shapes/script/generate_endpoint_simple_loss_query100x1_bank.py"
query_common=(
  --query-file "$query_file"
  --query-ids "$query_ids"
  --namespace "$query_namespace"
  --experiment "$experiment"
  --train-seed "$train_seed"
  --epochs 200
  --batch-size "${QUERY_BATCH_SIZE:-4}"
  --shard-count 2
)

echo "[query] endpoint simple-loss mean100t MC1; cached because reuse is allowed"
run_task 0 "${gpu_ids[0]}" \
  "$python_bin" "$query_bank" "${query_common[@]}" --shard-index 0 \
  >"$log_root/query_gpu_${gpu_ids[0]}.log" 2>&1 &
query_pid0=$!
run_task 1 "${gpu_ids[1]}" \
  "$python_bin" "$query_bank" "${query_common[@]}" --shard-index 1 \
  >"$log_root/query_gpu_${gpu_ids[1]}.log" 2>&1 &
query_pid1=$!
failed=0
wait "$query_pid0" || failed=1
wait "$query_pid1" || failed=1
(( failed == 0 )) || { echo "query worker failed; inspect $log_root" >&2; exit 1; }

ibrun -n 1 -o 0 env JAX_PLATFORMS=cpu \
  "$python_bin" "$query_bank" "${query_common[@]}" --merge-only \
  --artifact-list "$query_artifact_list" \
  >"$log_root/query_merge.log" 2>&1
mapfile -t query_artifacts < "$query_artifact_list"
[[ ${#query_artifacts[@]} == 100 ]] || {
  echo "expected 100 query artifacts, found ${#query_artifacts[@]}" >&2
  exit 1
}
query_joined="$(IFS=:; echo "${query_artifacts[*]}")"

echo "[train+score] mean100t MC1 gradient -> AdamW-full -> linear contraction -> discard train"
stage="$shapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py"
train_pids=()
for shard in 0 1; do
  gpu="${gpu_ids[$shard]}"
  (
    run_task "$shard" "$gpu" env \
      EXPERIMENT_TAG="$experiment" \
      TRAIN_SEED="$train_seed" \
      JAX_EPOCHS=200 \
      DATAPOINT_MODEL_MODE=prompted_solo \
      SAMPLE_MODEL_MODE=prompted_solo \
      ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo \
      QUERY=shape_cube,object_hue_0,wall_hue_0,floor_hue_0 \
      INITIAL_SEED=0 SAMPLE_SEED=0 \
      TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse \
      TRAJ_PARAMETER_SOURCE=raw \
      TRAJ_NUM_SNAPSHOTS=100 \
      TRAJ_TRAIN_MC_SAMPLES=1 \
      TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-2}" \
      TRAJ_TRACIN_PROJ_DIM=4096 \
      TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=1 \
      TRAJ_TRACIN_TRAIN_AGGREGATE_NUM_TIMESTEPS=100 \
      TRAJ_TRACIN_TRAIN_TIMESTAMP_CHUNK_SIZE=100 \
      TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32 \
      TRAJ_TRACIN_TRAIN_BATCH_MODE=vmap \
      TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adamw_residual_update \
      TRAJ_TRACIN_STAGE_MODE=train \
      TRAJ_TRACIN_CKPT_SHARD_INDEX="$shard" \
      TRAJ_TRACIN_CKPT_SHARD_COUNT=2 \
      TRAJ_TRACIN_TRAIN_EXCLUDE_FINAL_CHECKPOINT=1 \
      TRAJ_TRACIN_FUSED_STREAM_QUERY_ARTIFACTS="$query_joined" \
      TRAJ_TRACIN_FUSED_STREAM_OUTPUT="$component_prefix" \
      TRAJ_TRACIN_FUSED_SAVE_EVERY=1 \
      TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$log_root/no_train_artifact_gpu_${gpu}.npz" \
      TRAJ_TRACIN_SKIP_STAGE_MERGE=1 \
      "$python_bin" "$stage"
  ) >"$log_root/train_score_gpu_${gpu}.log" 2>&1 &
  train_pids+=("$!")
done
failed=0
for pid in "${train_pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "train/score worker failed; inspect $log_root" >&2; exit 1; }
for gpu in "${gpu_ids[@]}"; do
  [[ ! -e "$log_root/no_train_artifact_gpu_${gpu}.npz" ]] || {
    echo "invariant failed: fused run unexpectedly persisted a train artifact" >&2
    exit 1
  }
done

ibrun -n 1 -o 0 env JAX_PLATFORMS=cpu \
  "$python_bin" "$shapes/script/merge_fused_endpoint_tracin100x1_scores.py" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --component-prefix "$component_prefix" --namespace "$score_namespace" \
  --experiment "$experiment" --train-seed "$train_seed" --shard-count 2 \
  >"$log_root/merge.log" 2>&1

if [[ "${RUN_LDS:-1}" == "1" ]]; then
  ibrun -n 1 -o 0 env JAX_PLATFORMS=cpu \
    "$python_bin" "$shapes/script/run_traj_tracin_lds_cached.py" \
    --execute --experiment "$experiment" --train-seed "$train_seed" \
    --query-file "$query_file" --query-ids "$query_ids" \
    --score-schemes "$score_namespace" --prediction-sign -1 \
    >"$log_root/lds.log" 2>&1
fi

echo "[done] endpoint-TracIn AdamW-full train100x1/query100x1 linear; four variants; no train artifact"
