#!/usr/bin/env bash
#SBATCH -J 3d-a4-o8-1k
#SBATCH -o 3d-a4-o8-1k-%j.out
#SBATCH -e 3d-a4-o8-1k-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
query_runner="$shapes/script/run_traj_tracin_queries_and_scores.py"
score_driver="$shapes/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

exp="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
epochs="${JAX_EPOCHS:-200}"
probe_seeds=(20260917 20260918 73194261 418507293 90216487 563809241 247196803 816430927)
suffix="own8_1000t_no_alignment"
log_root="$shapes/result/$exp/logs/adamw_four_event_own8_1000t_no_alignment/${SLURM_JOB_ID}"
mkdir -p "$log_root"

query_patterns=""
probe_seed_csv=""
for probe_seed in "${probe_seeds[@]}"; do
  namespace="loss_direction_predicted_noise_probe1_timestamp_shared_seed${probe_seed}_checkpoint_own_trajectory_1000t_r0"
  [[ -z "$query_patterns" ]] || query_patterns+=','
  [[ -z "$probe_seed_csv" ]] || probe_seed_csv+=','
  query_patterns+="$namespace"
  probe_seed_csv+="$probe_seed"
done

echo "[phase 1/3] eight timestamp-shared probes on checkpoint-own trajectories; 1000 timestamps"
echo "[invariant] for fixed (probe,timestamp), direction is shared over all checkpoints and queries"
echo "[no alignment] every train event is combined with all 1000 query timestamps"
echo "[weighting] AdamW internal LR retained; outer checkpoint LR removed"

export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
pids=()
for gpu in 0 1; do
  (
    for slot in 0 1 2 3; do
      index=$((gpu * 4 + slot))
      probe_seed="${probe_seeds[$index]}"
      namespace="loss_direction_predicted_noise_probe1_timestamp_shared_seed${probe_seed}_checkpoint_own_trajectory_1000t_r0"
      echo "[gpu $gpu] probe $((index + 1))/8 seed=$probe_seed"
      python "$query_runner" --execute \
        --experiment "$exp" --train-seed "$train_seed" --epochs "$epochs" \
        --query-ids 0,1,2,3,4,5,6,7,8,9 --gpus "$gpu" \
        --skip-sampling --skip-score --artifact-namespace "$namespace" \
        --query-objective trajectory_predicted_noise_probe \
        --predicted-noise-probe-index 0 --predicted-noise-probe-count 1 \
        --predicted-noise-probe-mode timestamp_shared_gaussian \
        --predicted-noise-probe-seed "$probe_seed" \
        --num-snapshots 1000 --log-prefix "a4_o8_1k_seed${probe_seed}_gpu${gpu}"
    done
  ) >"$log_root/query_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "query generation failed; inspect $log_root" >&2; exit 1; }
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

echo "[phase 2/3] four AdamW train methods x square/root = eight score schemes"
schemes=""
for method in four e1 four_residual e1_residual; do
  for spec in 'squared l2_squared' 'probe_l2 probe_l2'; do
    read -r contraction reduction <<<"$spec"
    namespace_suffix="adamw4_${method}_${suffix}"
    run_id="${SLURM_JOB_ID}_${method}_${reduction}"
    pids=()
    for shard in 0 1; do
      (
        CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
          python "$score_driver" score-shard \
            --experiment "$exp" --train-seed "$train_seed" \
            --train-namespace "traj_tracin_adamw4_${method}" \
            --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
            --checkpoint-weighting uniform \
            --run-id "$run_id" --shard-index "$shard" --shard-count 2 \
            --num-probes 8 --contraction "$contraction" \
            --namespace-suffix "$namespace_suffix" \
            --query-namespace-patterns "$query_patterns" \
            --expected-query-probe-mode timestamp_shared_gaussian \
            --expected-query-probe-seeds "$probe_seed_csv"
      ) >"$log_root/score_${method}_${reduction}_shard_${shard}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || { echo "score failed: $method/$reduction; inspect $log_root" >&2; exit 1; }

    JAX_PLATFORMS=cpu python "$score_driver" merge \
      --experiment "$exp" --train-seed "$train_seed" \
      --train-namespace "traj_tracin_adamw4_${method}" \
      --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
      --checkpoint-weighting uniform \
      --run-id "$run_id" --shard-count 2 --num-probes 8 \
      --contraction "$contraction" --namespace-suffix "$namespace_suffix" \
      --query-namespace-patterns "$query_patterns" \
      --expected-query-probe-mode timestamp_shared_gaussian \
      --expected-query-probe-seeds "$probe_seed_csv" --expected-terms 49000

    scheme="predicted_noise_jvp_${reduction}_probe8_${namespace_suffix}"
    [[ -z "$schemes" ]] || schemes+=','
    schemes+="$scheme"
  done
done

echo "[phase 3/3] cached LDS for all eight schemes; fixed p1"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$exp" --train-seed "$train_seed" \
  --score-schemes "$schemes" --prediction-sign=1

echo "[done] own8 full-1000 non-aligned square/root scores and LDS"
