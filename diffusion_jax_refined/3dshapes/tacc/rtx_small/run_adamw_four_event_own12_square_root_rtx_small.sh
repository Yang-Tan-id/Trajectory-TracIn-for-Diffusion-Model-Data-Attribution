#!/usr/bin/env bash
#SBATCH -J 3d-adam4-o12
#SBATCH -o 3d-adam4-o12-%j.out
#SBATCH -e 3d-adam4-o12-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/materialize_adamw_four_event_train_parts.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
query_driver="$shapes/script/run_traj_tracin_queries_and_scores.py"
score_driver="$shapes/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

exp="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
epochs="${JAX_EPOCHS:-200}"
probe_seeds=(314159265 271828183 161803399 141421357 538102947 794615203 126937481 682450719 905173624 417286953 263590817 849031576)
probe_seeds_csv="$(IFS=,; echo "${probe_seeds[*]}")"
log_root="$shapes/result/$exp/logs/adamw_four_event_own12/${SLURM_JOB_ID}"
mkdir -p "$log_root"

patterns=()
for probe_seed in "${probe_seeds[@]}"; do
  patterns+=("loss_direction_predicted_noise_probe1_timestamp_shared_adamw_own_seed${probe_seed}_checkpoint_own_trajectory_r0")
done
query_patterns="$(IFS=,; echo "${patterns[*]}")"

echo "[phase 1/4] generate 12 timestamp-shared probes on every checkpoint's own trajectory"
echo "[invariant] v[r,c,t] = v[r,t]; x[c,t] is checkpoint c's own trajectory state"
export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
pids=()
for gpu in 0 1; do
  (
    for slot in 0 1 2 3 4 5; do
      probe=$((gpu * 6 + slot))
      probe_seed="${probe_seeds[$probe]}"
      namespace="${patterns[$probe]}"
      echo "[gpu $gpu] P$((probe + 1))/12 seed=$probe_seed namespace=$namespace"
      python "$query_driver" \
          --execute --experiment "$exp" --train-seed "$seed" --epochs "$epochs" \
          --query-ids 0,1,2,3,4,5,6,7,8,9 --gpus "$gpu" \
          --skip-sampling --skip-score --artifact-namespace "$namespace" \
          --query-objective trajectory_predicted_noise_probe \
          --predicted-noise-probe-index 0 --predicted-noise-probe-count 1 \
          --predicted-noise-probe-mode timestamp_shared_gaussian \
          --predicted-noise-probe-seed "$probe_seed" --num-snapshots 10 \
          --log-prefix "adamw_own12_p$((probe + 1))_seed${probe_seed}_gpu${gpu}"
    done
  ) >"$log_root/query_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY
(( failed == 0 )) || { echo "own-trajectory query generation failed; inspect $log_root" >&2; exit 1; }

echo "[phase 2/4] materialize no-timestep-alignment AdamW train banks"
python "$shapes/script/materialize_adamw_four_event_train_parts.py" \
  --experiment "$exp" --train-seed "$seed" --attribution-points "${ATTRIBUTION_POINTS:-5000}"

echo "[phase 3/4] four train methods x square/root combine"
schemes=""
for method in four e1 four_residual e1_residual; do
  for spec in 'squared l2_squared' 'probe_l2 probe_l2'; do
    read -r contraction reduction <<<"$spec"
    suffix="adamw4_${method}_own12"
    run_id="${SLURM_JOB_ID}_${method}_${reduction}"
    pids=()
    for shard in 0 1; do
      (
        CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
          python "$score_driver" score-shard \
            --experiment "$exp" --train-seed "$seed" \
            --train-namespace "traj_tracin_adamw4_${method}" \
            --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
            --run-id "$run_id" --shard-index "$shard" --shard-count 2 \
            --num-probes 12 --contraction "$contraction" --namespace-suffix "$suffix" \
            --query-namespace-patterns "$query_patterns" \
            --expected-query-probe-mode timestamp_shared_gaussian \
            --expected-query-probe-seeds "$probe_seeds_csv"
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
      --run-id "$run_id" --shard-count 2 --num-probes 12 \
      --contraction "$contraction" --namespace-suffix "$suffix" \
      --query-namespace-patterns "$query_patterns" \
      --expected-query-probe-mode timestamp_shared_gaussian \
      --expected-query-probe-seeds "$probe_seeds_csv" --expected-terms 490
    scheme="predicted_noise_jvp_${reduction}_probe12_${suffix}"
    [[ -z "$schemes" ]] || schemes+=','
    schemes+="$scheme"
  done
done

echo "[phase 4/4] cached LDS; fixed p1"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$exp" --train-seed "$seed" \
  --score-schemes "$schemes" --prediction-sign=1
echo "[done] AdamW four-event own12 square/root LDS"
