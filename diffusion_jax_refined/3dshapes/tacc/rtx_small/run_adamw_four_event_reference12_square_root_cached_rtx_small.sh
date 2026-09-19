#!/usr/bin/env bash
#SBATCH -J 3d-adam4-p12
#SBATCH -o 3d-adam4-p12-%j.out
#SBATCH -e 3d-adam4-p12-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 16:00:00
set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/materialize_adamw_four_event_train_parts.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"; source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh; conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
exp="${EXPERIMENT_TAG:-experiment1}"; seed="${TRAIN_SEED:-42}"; driver="$shapes/script/run_predicted_noise_jvp_l2_squared.py"
python "$shapes/script/materialize_adamw_four_event_train_parts.py" --experiment "$exp" --train-seed "$seed" --attribution-points "${ATTRIBUTION_POINTS:-5000}"
seeds='314159265,271828183,161803399,141421357,538102947,794615203,126937481,682450719,905173624,417286953,263590817,849031576'
patterns=''; IFS=',' read -ra aa <<<"$seeds"; for x in "${aa[@]}"; do [[ -z "$patterns" ]] || patterns+=','; patterns+="loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${x}_r0"; done
schemes=''
for method in four e1 four_residual e1_residual; do
 for spec in 'squared l2_squared' 'probe_l2 probe_l2'; do
  read -r contraction reduction <<<"$spec"; suffix="adamw4_${method}_reference12"; run="${SLURM_JOB_ID}_${method}_${reduction}"; pids=()
  for shard in 0 1; do
   (CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python "$driver" score-shard --experiment "$exp" --train-seed "$seed" --train-namespace "traj_tracin_adamw4_${method}" --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment --run-id "$run" --shard-index "$shard" --shard-count 2 --num-probes 12 --contraction "$contraction" --namespace-suffix "$suffix" --query-namespace-patterns "$patterns" --expected-query-probe-mode timestamp_shared_gaussian --expected-query-probe-seeds "$seeds") & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  JAX_PLATFORMS=cpu python "$driver" merge --experiment "$exp" --train-seed "$seed" --train-namespace "traj_tracin_adamw4_${method}" --train-feature-semantics fixed_checkpoint_adamw_event_no_timestamp_alignment --run-id "$run" --shard-count 2 --num-probes 12 --contraction "$contraction" --namespace-suffix "$suffix" --query-namespace-patterns "$patterns" --expected-query-probe-mode timestamp_shared_gaussian --expected-query-probe-seeds "$seeds" --expected-terms 490
  name="predicted_noise_jvp_${reduction}_probe12_${suffix}"; [[ -z "$schemes" ]] || schemes+=','; schemes+="$name"
 done
done
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" --execute --experiment "$exp" --train-seed "$seed" --score-schemes "$schemes" --prediction-sign=1
echo "[done] AdamW four-event reference12 square/root LDS"
