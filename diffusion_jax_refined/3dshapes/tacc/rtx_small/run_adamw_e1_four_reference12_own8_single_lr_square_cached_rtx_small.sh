#!/usr/bin/env bash
#SBATCH -J 3d-a4-1lr-sq
#SBATCH -o 3d-a4-1lr-sq-%j.out
#SBATCH -e 3d-a4-1lr-sq-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
driver="$shapes/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

exp="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
log_root="$shapes/result/$exp/logs/adamw_e1_four_reference12_own8_single_lr_square/$SLURM_JOB_ID"
mkdir -p "$log_root"

ref_seeds='314159265,271828183,161803399,141421357,538102947,794615203,126937481,682450719,905173624,417286953,263590817,849031576'
ref_patterns=''
IFS=',' read -ra ref_seed_array <<<"$ref_seeds"
for probe_seed in "${ref_seed_array[@]}"; do
  [[ -z "$ref_patterns" ]] || ref_patterns+=','
  ref_patterns+="loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${probe_seed}_r0"
done

own_seeds='20260917,20260918,73194261,418507293,90216487,563809241,247196803,816430927'
own_patterns='loss_direction_predicted_noise_probe1_timestamp_shared_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed20260918_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed73194261_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed418507293_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed90216487_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed563809241_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed247196803_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed816430927_checkpoint_own_trajectory_r0'

schemes=''
run_bank() {
  local bank="$1" num_probes="$2" patterns="$3" probe_seeds="$4"
  local method suffix run_id shard pid failed scheme
  for method in e1 four; do
    suffix="adamw4_${method}_${bank}_single_lr"
    run_id="${SLURM_JOB_ID}_${bank}_${method}"
    echo "[score] bank=$bank method=$method: eta * square(z / eta)"
    pids=()
    for shard in 0 1; do
      (
        CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
          python "$driver" score-shard \
            --experiment "$exp" --train-seed "$train_seed" \
            --train-namespace "traj_tracin_adamw4_${method}" \
            --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
            --checkpoint-weighting inverse_stored_lr \
            --run-id "$run_id" --shard-index "$shard" --shard-count 2 \
            --num-probes "$num_probes" --contraction squared \
            --namespace-suffix "$suffix" \
            --query-namespace-patterns "$patterns" \
            --expected-query-probe-mode timestamp_shared_gaussian \
            --expected-query-probe-seeds "$probe_seeds"
      ) >"$log_root/${bank}_${method}_shard_${shard}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || {
      echo "score failed: bank=$bank method=$method; inspect $log_root" >&2
      exit 1
    }
    JAX_PLATFORMS=cpu python "$driver" merge \
      --experiment "$exp" --train-seed "$train_seed" \
      --train-namespace "traj_tracin_adamw4_${method}" \
      --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
      --checkpoint-weighting inverse_stored_lr \
      --run-id "$run_id" --shard-count 2 --num-probes "$num_probes" \
      --contraction squared --namespace-suffix "$suffix" \
      --query-namespace-patterns "$patterns" \
      --expected-query-probe-mode timestamp_shared_gaussian \
      --expected-query-probe-seeds "$probe_seeds" --expected-terms 490
    scheme="predicted_noise_jvp_l2_squared_probe${num_probes}_${suffix}"
    [[ -z "$schemes" ]] || schemes+=','
    schemes+="$scheme"
  done
}

echo '[definition] remove AdamW internal LR before square, then apply the checkpoint LR exactly once'
echo '[methods] non-residual E1 and FOUR only; p1; 10 timestamps; four normalization variants'
run_bank reference12 12 "$ref_patterns" "$ref_seeds"
run_bank own8 8 "$own_patterns" "$own_seeds"

echo '[LDS] four new scores x four targets x four normalization variants x ten queries'
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$exp" --train-seed "$train_seed" \
  --score-schemes "$schemes" --prediction-sign=1

echo "[done] schemes=$schemes"
