#!/usr/bin/env bash
#SBATCH -J 3d-djvp-r10-1lr
#SBATCH -o 3d-djvp-r10-1lr-%j.out
#SBATCH -e 3d-djvp-r10-1lr-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00

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
cached="$shapes/result/$exp/direct_jvp/reference_constant_lr"

echo '[cached] direct JVP reference trajectory; 10 timestamps; reuse 16 H100 shard partials'
echo '[definition] checkpoint contribution = square(JVP(adamw_direction)) / checkpoint_lr'
echo '[merge] recover checkpoint increments from adjacent cumulative partials'

schemes="$(JAX_PLATFORMS=cpu python "$driver" merge \
  --experiment "$exp" \
  --train-seed "$seed" \
  --trajectory reference \
  --out-dir "$cached" \
  --num-shards 16 \
  --checkpoint-weighting single_lr | tail -n 1)"

echo "[schemes] $schemes"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "$exp" \
  --train-seed "$seed" \
  --score-schemes "$schemes" \
  --prediction-sign=1

echo '[done] direct-JVP reference 10t square with one checkpoint LR'
