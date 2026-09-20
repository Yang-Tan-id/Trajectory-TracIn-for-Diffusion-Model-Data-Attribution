#!/usr/bin/env bash
#SBATCH -J 3d-a12-subsets
#SBATCH -o 3d-a12-subsets-%j.out
#SBATCH -e 3d-a12-subsets-%j.err
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
analyzer="$shapes/script/analyze_predicted_noise_probe8_all_subset_sizes.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

exp="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
num_probes=12
seeds='314159265,271828183,161803399,141421357,538102947,794615203,126937481,682450719,905173624,417286953,263590817,849031576'
patterns=''
IFS=',' read -ra seed_array <<<"$seeds"
for probe_seed in "${seed_array[@]}"; do
  [[ -z "$patterns" ]] || patterns+=','
  patterns+="loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${probe_seed}_r0"
done

suffix='adamw4_four_reference12_no_outer_lr_all_subsets'
score_namespace="traj_tracin_predicted_noise_jvp_termwise_squared_per_probe_probe12_${suffix}"
log_root="$shapes/result/$exp/logs/adamw_reference12_four_termwise_square_all_subsets/$SLURM_JOB_ID"
mkdir -p "$log_root"

echo '[phase 1/2] retain all 12 per-probe termwise-square scores; AdamW FOUR, reference trajectory, uniform checkpoint weights'
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "$driver" score-shard \
        --experiment "$exp" --train-seed "$train_seed" \
        --train-namespace traj_tracin_adamw4_four \
        --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment \
        --checkpoint-weighting uniform \
        --run-id "$SLURM_JOB_ID" --shard-index "$shard" --shard-count 2 \
        --num-probes "$num_probes" --contraction termwise_squared_per_probe \
        --namespace-suffix "$suffix" --query-namespace-patterns "$patterns" \
        --expected-query-probe-mode timestamp_shared_gaussian \
        --expected-query-probe-seeds "$seeds"
  ) >"$log_root/score_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "Score shard failed; inspect $log_root" >&2; exit 1; }

echo '[phase 2/2] enumerate all 4095 subsets, compute LDS, and draw every point with mean/quartiles'
JAX_PLATFORMS=cpu python "$analyzer" \
  --experiment "$exp" --train-seed "$train_seed" \
  --run-id "$SLURM_JOB_ID" --num-probes "$num_probes" \
  --expected-terms 490 \
  --prediction-sign=1 \
  --score-namespace "$score_namespace" \
  --analysis-label adamw_reference12_four_termwise_square_all_subsets \
  --score-label 'AdamW FOUR / reference12 / termwise-square / p1'

echo "[done] output: $shapes/result/$exp/eval/adamw_reference12_four_termwise_square_all_subsets/run_$SLURM_JOB_ID"
