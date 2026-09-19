#!/usr/bin/env bash
#SBATCH -J 3d-adam4-o8n
#SBATCH -o 3d-adam4-o8n-%j.out
#SBATCH -e 3d-adam4-o8n-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/materialize_adamw_four_event_train_parts.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
score_driver="$shapes/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exp="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
probe_seeds="20260917,20260918,73194261,418507293,90216487,563809241,247196803,816430927"
query_patterns="loss_direction_predicted_noise_probe1_timestamp_shared_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed20260918_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed73194261_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed418507293_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed90216487_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed563809241_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed247196803_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed816430927_checkpoint_own_trajectory_r0"
log_root="$shapes/result/$exp/logs/adamw_four_event_own8_no_outer_lr/${SLURM_JOB_ID}"
mkdir -p "$log_root"

echo "[phase 1/3] reuse eight completed timestamp-shared own-trajectory probe banks"
echo "[invariant] v[r,c,t] = v[r,t]; x[c,t] is checkpoint c's own trajectory state"
echo "[weighting] AdamW internal LR retained; outer checkpoint LR removed; timestamps retain weight 1/10"

echo "[phase 2/3] four train methods x square/root combine"
schemes=""
for method in four e1 four_residual e1_residual; do
  for spec in 'squared l2_squared' 'probe_l2 probe_l2'; do
    read -r contraction reduction <<<"$spec"
    suffix="adamw4_${method}_own8_no_outer_lr"
    run_id="${SLURM_JOB_ID}_${method}_${reduction}"
    pids=()
    for shard in 0 1; do
      (
        CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
          python "$score_driver" score-shard \
            --experiment "$exp" --train-seed "$seed" \
            --train-namespace "traj_tracin_adamw4_${method}" \
            --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
            --checkpoint-weighting uniform \
            --run-id "$run_id" --shard-index "$shard" --shard-count 2 \
            --num-probes 8 --contraction "$contraction" --namespace-suffix "$suffix" \
            --query-namespace-patterns "$query_patterns" \
            --expected-query-probe-mode timestamp_shared_gaussian \
            --expected-query-probe-seeds "$probe_seeds"
      ) >"$log_root/score_${method}_${reduction}_shard_${shard}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || { echo "score failed: $method/$reduction; inspect $log_root" >&2; exit 1; }
    JAX_PLATFORMS=cpu python "$score_driver" merge \
      --experiment "$exp" --train-seed "$seed" \
      --train-namespace "traj_tracin_adamw4_${method}" \
      --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
      --checkpoint-weighting uniform \
      --run-id "$run_id" --shard-count 2 --num-probes 8 \
      --contraction "$contraction" --namespace-suffix "$suffix" \
      --query-namespace-patterns "$query_patterns" \
      --expected-query-probe-mode timestamp_shared_gaussian \
      --expected-query-probe-seeds "$probe_seeds" --expected-terms 490
    scheme="predicted_noise_jvp_${reduction}_probe8_${suffix}"
    [[ -z "$schemes" ]] || schemes+=','
    schemes+="$scheme"
  done
done

echo "[phase 3/3] cached LDS; fixed p1"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$exp" --train-seed "$seed" \
  --score-schemes "$schemes" --prediction-sign=1
echo "[done] AdamW four-event cached own8 square/root without outer checkpoint LR"
