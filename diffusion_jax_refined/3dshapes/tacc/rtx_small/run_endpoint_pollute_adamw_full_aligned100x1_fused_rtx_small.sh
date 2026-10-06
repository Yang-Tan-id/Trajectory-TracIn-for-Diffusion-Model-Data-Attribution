#!/usr/bin/env bash
#SBATCH -J 3d-ep100-fused
#SBATCH -o 3d-ep100-fused-%j.out
#SBATCH -e 3d-ep100-fused-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/generate_polluted_delta_query_bank_checkpoint_major.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
experiment="${EXPERIMENT_TAG:-experiment1}"; seed="${TRAIN_SEED:-42}"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_namespace="recreate_q0_99_polluted_endpoint_next_delta_l2normalized_100t"
raw_namespace="recreate_q0_99_polluted_endpoint_next_delta_raw_100t"
log_root="$shapes/result/$experiment/logs/endpoint_pollute_adamw_full_aligned100x1_fused/${SLURM_JOB_ID}"
component_prefix="$log_root/components"
mkdir -p "$log_root"
export PYTHONUNBUFFERED=1 TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}" XLA_PYTHON_CLIENT_PREALLOCATE=false

bank="$shapes/script/generate_polluted_delta_query_bank_checkpoint_major.py"
common=(--query-file "$query_file" --query-ids "$query_ids" --experiment "$experiment" --train-seed "$seed" --epochs 200 --raw-namespace "$raw_namespace" --normalized-namespace "$query_namespace" --timestamp-count 100 --normalized-only --batch-size "${QUERY_BATCH_SIZE:-2}" --shard-count 2)
for gpu in 0 1; do CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 "$python_bin" "$bank" "${common[@]}" --shard-index $gpu >"$log_root/query_gpu_${gpu}.log" 2>&1 & pids[$gpu]=$!; done
for pid in "${pids[@]}"; do wait "$pid"; done
JAX_PLATFORMS=cpu "$python_bin" "$bank" "${common[@]}" --merge-only >"$log_root/query_merge.log" 2>&1

mapfile -t query_artifacts < <(find "$shapes/result/$experiment/sample_ddim_eta0_1000" -type f -path "*_query_gradient_${query_namespace}/traj_tracin/query_gradient_artifact.npz" | sort)
[[ ${#query_artifacts[@]} == 100 ]] || { echo "expected 100 query artifacts, found ${#query_artifacts[@]}" >&2; exit 1; }
query_joined="$(IFS=:; echo "${query_artifacts[*]}")"

stage="$shapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py"
for gpu in 0 1; do
  (export CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 EXPERIMENT_TAG="$experiment" TRAIN_SEED="$seed" JAX_EPOCHS=200 DATAPOINT_MODEL_MODE=prompted_solo SAMPLE_MODEL_MODE=prompted_solo ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo QUERY=shape_cube,object_hue_0,wall_hue_0,floor_hue_0 INITIAL_SEED=0 SAMPLE_SEED=0 TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse TRAJ_PARAMETER_SOURCE=raw TRAJ_NUM_SNAPSHOTS=100 TRAJ_TRAIN_MC_SAMPLES=1 TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-16}" TRAJ_TRACIN_PROJ_DIM=4096 TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0 TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32 TRAJ_TRACIN_TRAIN_BATCH_MODE=vmap TRAJ_TRACIN_TRAIN_NOISE_MODE=checkpoint_timestamp_shared TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adamw_residual_update TRAJ_TRACIN_STAGE_MODE=train TRAJ_TRACIN_CKPT_SHARD_INDEX=$gpu TRAJ_TRACIN_CKPT_SHARD_COUNT=2 TRAJ_TRACIN_TRAIN_EXCLUDE_FINAL_CHECKPOINT=1 TRAJ_TRACIN_FUSED_STREAM_QUERY_ARTIFACTS="$query_joined" TRAJ_TRACIN_FUSED_STREAM_OUTPUT="$component_prefix"; "$python_bin" "$stage") >"$log_root/train_score_gpu_${gpu}.log" 2>&1 & pids[$gpu]=$!
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$python_bin" "$shapes/script/merge_fused_endpoint_pollute_scores.py" --query-file "$query_file" --query-ids "$query_ids" --component-prefix "$component_prefix" --experiment "$experiment" --train-seed "$seed"
schemes=""
for reduction in linear termwise_squared timestamp_sum_squared; do schemes+=" recreate_adamw_full_polluted_endpoint_delta_l2normalized_${reduction}_aligned100x1_q0_99"; done
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$python_bin" "$shapes/script/run_traj_tracin_lds_cached.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$query_ids" --score-schemes "$schemes" --prediction-sign 1 >"$log_root/lds.log" 2>&1
echo "[done] fused endpoint-pollute AdamW-full aligned100x1: 3 reductions x 4 variants x 100 queries"
