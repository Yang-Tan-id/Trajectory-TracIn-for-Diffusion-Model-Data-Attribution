#!/usr/bin/env bash
#SBATCH -J 3d-djvp-r100-h100
#SBATCH -o 3d-djvp-r100-h100-%j.out
#SBATCH -e 3d-djvp-r100-h100-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
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
out="$shapes/result/$exp/direct_jvp/reference100t_constant_lr"
num_shards=16
gpu_per_node=4

echo '[definition] direct JVP; reference trajectory; 100 timestamps; constant checkpoint weights; projection=4096'
echo '[parallel] four H100 nodes x four GPUs; 16 shards; 312-313 attribution datapoints per GPU; walltime=48h'

run_slot() {
  local slot="$1"
  shift
  local gpu="$((slot % gpu_per_node))"
  ibrun -n 1 -o "$slot" \
    env CUDA_VISIBLE_DEVICES="$gpu" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
    "$@"
}

pids=()
for shard in $(seq 0 15); do
  (
    run_slot "$shard" python "$driver" score-shard \
      --experiment "$exp" \
      --train-seed "$seed" \
      --trajectory reference \
      --num-timestamps 100 \
      --out-dir "$out" \
      --shard-id "$shard" \
      --num-shards "$num_shards"
  ) >"direct-jvp-reference100t-h100-gpu-${shard}-${SLURM_JOB_ID}.log" 2>&1 &
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
  --trajectory reference \
  --num-timestamps 100 \
  --out-dir "$out" \
  --num-shards "$num_shards" | tail -n 1)"

JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "$exp" \
  --train-seed "$seed" \
  --score-schemes "$schemes" \
  --prediction-sign=1

echo '[done] direct-JVP reference100t constant-LR eight scores'
