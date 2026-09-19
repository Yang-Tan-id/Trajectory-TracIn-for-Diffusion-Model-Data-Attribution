#!/usr/bin/env bash
#SBATCH -J 3d-a12-evsq
#SBATCH -o 3d-a12-evsq-%j.out
#SBATCH -e 3d-a12-evsq-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/materialize_adamw_four_event_train_parts.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
exp="${EXPERIMENT_TAG:-experiment1}"; seed="${TRAIN_SEED:-42}"; driver="$shapes/script/run_predicted_noise_jvp_l2_squared.py"
python "$shapes/script/materialize_adamw_four_event_train_parts.py" --experiment "$exp" --train-seed "$seed" --attribution-points "${ATTRIBUTION_POINTS:-5000}"
seeds='314159265,271828183,161803399,141421357,538102947,794615203,126937481,682450719,905173624,417286953,263590817,849031576'
patterns=''; IFS=',' read -ra aa <<<"$seeds"; for x in "${aa[@]}"; do [[ -z "$patterns" ]] || patterns+=','; patterns+="loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${x}_r0"; done

score_event() {
  local method="$1" suffix="adamw4_${1}_reference12" run="${SLURM_JOB_ID}_${1}_l2_squared" pids=()
  local storage_namespace="traj_tracin_predicted_noise_jvp_l2_squared_probe12_${suffix}"
  local completed
  completed="$(find "$shapes/result/$exp/attribution_score/prompted_solo/train_seed_${seed}" -path "*/${storage_namespace}/*/scores.npy" 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$completed" -ge 40 ]]; then
    echo "[skip] complete cached event score: method=$method files=$completed"
    return
  fi
  for shard in 0 1; do
    (CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python "$driver" score-shard --experiment "$exp" --train-seed "$seed" --train-namespace "traj_tracin_adamw4_${method}" --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment --run-id "$run" --shard-index "$shard" --shard-count 2 --num-probes 12 --contraction squared --namespace-suffix "$suffix" --query-namespace-patterns "$patterns" --expected-query-probe-mode timestamp_shared_gaussian --expected-query-probe-seeds "$seeds") & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  JAX_PLATFORMS=cpu python "$driver" merge --experiment "$exp" --train-seed "$seed" --train-namespace "traj_tracin_adamw4_${method}" --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment --run-id "$run" --shard-count 2 --num-probes 12 --contraction squared --namespace-suffix "$suffix" --query-namespace-patterns "$patterns" --expected-query-probe-mode timestamp_shared_gaussian --expected-query-probe-seeds "$seeds" --expected-terms 490
}

for method in e1 e2 e3 e4 e1_residual e2_residual e3_residual e4_residual; do score_event "$method"; done

raw_inputs='predicted_noise_jvp_l2_squared_probe12_adamw4_e1_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e2_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e3_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e4_reference12'
res_inputs='predicted_noise_jvp_l2_squared_probe12_adamw4_e1_residual_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e2_residual_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e3_residual_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e4_residual_reference12'
raw_output='predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_square_sum_reference12'
res_output='predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_residual_square_sum_reference12'
python "$shapes/script/combine_eventwise_squared_scores.py" --experiment "$exp" --train-seed "$seed" --input-namespaces "$raw_inputs" --output-namespace "$raw_output"
python "$shapes/script/combine_eventwise_squared_scores.py" --experiment "$exp" --train-seed "$seed" --input-namespaces "$res_inputs" --output-namespace "$res_output"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" --execute --experiment "$exp" --train-seed "$seed" --score-schemes "$raw_output,$res_output" --prediction-sign=1
echo "[done] reference12 eventwise E1^2+E2^2+E3^2+E4^2 LDS"
