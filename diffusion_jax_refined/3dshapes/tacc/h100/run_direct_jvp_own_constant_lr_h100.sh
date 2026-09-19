#!/usr/bin/env bash
#SBATCH -J 3d-djvp-own-h100
#SBATCH -o 3d-djvp-own-h100-%j.out
#SBATCH -e 3d-djvp-own-h100-%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_direct_predicted_noise_jvp_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exp="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
driver="$shapes/script/run_direct_predicted_noise_jvp_scores.py"
out="$shapes/result/$exp/direct_jvp/own_constant_lr"

echo "[definition] direct JVP; checkpoint-own trajectory; 10 timestamps; constant checkpoint weights; projection=4096"
echo "[parallel] four H100 GPUs; four shards; 1250 attribution datapoints per GPU"
pids=()
for shard in 0 1 2 3; do
  (
    CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "$driver" score-shard \
        --experiment "$exp" \
        --train-seed "$seed" \
        --trajectory own \
        --out-dir "$out" \
        --shard-id "$shard" \
        --num-shards 4
  ) >"direct-jvp-own-h100-gpu-${shard}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || exit 1

schemes="$(JAX_PLATFORMS=cpu python "$driver" merge \
  --experiment "$exp" \
  --train-seed "$seed" \
  --trajectory own \
  --out-dir "$out" \
  --num-shards 4 | tail -n 1)"

JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "$exp" \
  --train-seed "$seed" \
  --score-schemes "$schemes" \
  --prediction-sign=1

echo "[done] direct-JVP own-trajectory constant-LR eight scores"
