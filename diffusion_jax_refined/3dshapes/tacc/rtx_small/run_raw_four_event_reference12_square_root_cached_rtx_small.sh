#!/usr/bin/env bash
#SBATCH -J 3d-raw4-ref12
#SBATCH -o 3d-raw4-ref12-%j.out
#SBATCH -e 3d-raw4-ref12-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/materialize_raw_four_event_train_parts.py" ]]; do
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
probe_seeds="314159265,271828183,161803399,141421357,538102947,794615203,126937481,682450719,905173624,417286953,263590817,849031576"
patterns=""
IFS=',' read -ra seed_array <<<"$probe_seeds"
for probe_seed in "${seed_array[@]}"; do
  [[ -z "$patterns" ]] || patterns+=','
  patterns+="loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${probe_seed}_r0"
done
log_root="$shapes/result/$exp/logs/raw_four_event_reference12/${SLURM_JOB_ID}"
mkdir -p "$log_root"

echo "[phase 1/3] materialize raw E1/four banks; no timestep alignment"
python "$shapes/script/materialize_raw_four_event_train_parts.py" \
  --experiment "$exp" --train-seed "$seed" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}"

echo "[phase 2/3] FOUR/E1 x square/root with cached reference12 probes"
schemes=""
for method in four e1; do
  for spec in 'squared l2_squared' 'probe_l2 probe_l2'; do
    read -r contraction reduction <<<"$spec"
    suffix="raw4_${method}_reference12"
    run_id="${SLURM_JOB_ID}_${method}_${reduction}"
    pids=()
    for shard in 0 1; do
      (
        CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
          python "$score_driver" score-shard \
            --experiment "$exp" --train-seed "$seed" \
            --train-namespace "traj_tracin_raw4_${method}" \
            --train-feature-semantics fixed_checkpoint_raw_event_gradient_no_timestamp_alignment \
            --run-id "$run_id" --shard-index "$shard" --shard-count 2 \
            --num-probes 12 --contraction "$contraction" \
            --namespace-suffix "$suffix" --query-namespace-patterns "$patterns" \
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
      --train-namespace "traj_tracin_raw4_${method}" \
      --train-feature-semantics fixed_checkpoint_raw_event_gradient_no_timestamp_alignment \
      --run-id "$run_id" --shard-count 2 --num-probes 12 \
      --contraction "$contraction" --namespace-suffix "$suffix" \
      --query-namespace-patterns "$patterns" \
      --expected-query-probe-mode timestamp_shared_gaussian \
      --expected-query-probe-seeds "$probe_seeds" --expected-terms 490
    scheme="predicted_noise_jvp_${reduction}_probe12_${suffix}"
    [[ -z "$schemes" ]] || schemes+=','
    schemes+="$scheme"
  done
done

echo "[phase 3/3] cached LDS; fixed p1"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$exp" --train-seed "$seed" \
  --score-schemes "$schemes" --prediction-sign=1
echo "[done] raw four-event reference12 square/root LDS"
