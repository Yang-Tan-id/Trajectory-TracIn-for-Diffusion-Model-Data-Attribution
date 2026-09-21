#!/usr/bin/env bash
#SBATCH -J 3d-a12-ev1lr
#SBATCH -o 3d-a12-ev1lr-%j.out
#SBATCH -e 3d-a12-ev1lr-%j.err
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
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exp="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
driver="$shapes/script/run_predicted_noise_jvp_l2_squared.py"

python "$shapes/script/materialize_adamw_four_event_train_parts.py" \
  --experiment "$exp" \
  --train-seed "$seed" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}"

seeds='314159265,271828183,161803399,141421357,538102947,794615203,126937481,682450719,905173624,417286953,263590817,849031576'
patterns=''
IFS=',' read -ra probe_seeds <<<"$seeds"
for probe_seed in "${probe_seeds[@]}"; do
  [[ -z "$patterns" ]] || patterns+=','
  patterns+="loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${probe_seed}_r0"
done

score_event() {
  local method="$1"
  local suffix="adamw4_${method}_reference12_single_lr"
  local run="${SLURM_JOB_ID}_${method}_l2_squared_single_lr"
  local storage_namespace="traj_tracin_predicted_noise_jvp_l2_squared_probe12_${suffix}"
  local completed
  local pids=()

  completed="$(find "$shapes/result/$exp/attribution_score/prompted_solo/train_seed_${seed}" \
    -path "*/${storage_namespace}/*/scores.npy" 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$completed" -ge 40 ]]; then
    echo "[skip] complete cached single-LR event score: method=$method files=$completed"
    return
  fi

  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "$driver" score-shard \
        --experiment "$exp" \
        --train-seed "$seed" \
        --train-namespace "traj_tracin_adamw4_${method}" \
        --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
        --checkpoint-weighting inverse_stored_lr \
        --run-id "$run" \
        --shard-index "$shard" \
        --shard-count 2 \
        --num-probes 12 \
        --contraction squared \
        --namespace-suffix "$suffix" \
        --query-namespace-patterns "$patterns" \
        --expected-query-probe-mode timestamp_shared_gaussian \
        --expected-query-probe-seeds "$seeds"
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  JAX_PLATFORMS=cpu python "$driver" merge \
    --experiment "$exp" \
    --train-seed "$seed" \
    --train-namespace "traj_tracin_adamw4_${method}" \
    --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
    --checkpoint-weighting inverse_stored_lr \
    --run-id "$run" \
    --shard-count 2 \
    --num-probes 12 \
    --contraction squared \
    --namespace-suffix "$suffix" \
    --query-namespace-patterns "$patterns" \
    --expected-query-probe-mode timestamp_shared_gaussian \
    --expected-query-probe-seeds "$seeds" \
    --expected-terms 490
}

echo '[definition] eta_c * sum_E mean_probe[(J u_E / eta_c)^2]'
echo '[phase 1/2] score E1-E4 and E1-E4 residual with inverse_stored_lr'
for method in e1 e2 e3 e4 e1_residual e2_residual e3_residual e4_residual; do
  score_event "$method"
done

raw_inputs='predicted_noise_jvp_l2_squared_probe12_adamw4_e1_reference12_single_lr,predicted_noise_jvp_l2_squared_probe12_adamw4_e2_reference12_single_lr,predicted_noise_jvp_l2_squared_probe12_adamw4_e3_reference12_single_lr,predicted_noise_jvp_l2_squared_probe12_adamw4_e4_reference12_single_lr'
residual_inputs='predicted_noise_jvp_l2_squared_probe12_adamw4_e1_residual_reference12_single_lr,predicted_noise_jvp_l2_squared_probe12_adamw4_e2_residual_reference12_single_lr,predicted_noise_jvp_l2_squared_probe12_adamw4_e3_residual_reference12_single_lr,predicted_noise_jvp_l2_squared_probe12_adamw4_e4_residual_reference12_single_lr'
raw_output='predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_square_sum_reference12_single_lr'
residual_output='predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_residual_square_sum_reference12_single_lr'

echo '[phase 2/2] sum E1^2+E2^2+E3^2+E4^2, then evaluate LDS'
python "$shapes/script/combine_eventwise_squared_scores.py" \
  --experiment "$exp" --train-seed "$seed" \
  --input-namespaces "$raw_inputs" --output-namespace "$raw_output"
python "$shapes/script/combine_eventwise_squared_scores.py" \
  --experiment "$exp" --train-seed "$seed" \
  --input-namespaces "$residual_inputs" --output-namespace "$residual_output"

JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "$exp" \
  --train-seed "$seed" \
  --score-schemes "$raw_output,$residual_output" \
  --prediction-sign=1

echo '[done] reference12 eventwise square-sum LDS with one checkpoint LR'
