#!/usr/bin/env bash
#SBATCH -J 3d-djvp-own
#SBATCH -o 3d-djvp-own-%j.out
#SBATCH -e 3d-djvp-own-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 16:00:00
set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_direct_predicted_noise_jvp_scores.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"; source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh; conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
exp="${EXPERIMENT_TAG:-experiment1}"; seed="${TRAIN_SEED:-42}"; driver="$shapes/script/run_direct_predicted_noise_jvp_scores.py"
out="$shapes/result/$exp/direct_jvp/own_constant_lr"
echo "[definition] direct JVP; checkpoint-own trajectory; constant checkpoint weights; projection=4096; resumable by checkpoint"
pids=()
for shard in 0 1; do
 (CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python "$driver" score-shard --experiment "$exp" --train-seed "$seed" --trajectory own --out-dir "$out" --shard-id "$shard" --num-shards 2) >"direct-jvp-own-gpu-${shard}-${SLURM_JOB_ID}.log" 2>&1 & pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; (( failed == 0 )) || exit 1
schemes="$(JAX_PLATFORMS=cpu python "$driver" merge --experiment "$exp" --train-seed "$seed" --trajectory own --out-dir "$out" --num-shards 2 | tail -n 1)"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" --execute --experiment "$exp" --train-seed "$seed" --score-schemes "$schemes" --prediction-sign=1
echo "[done] direct-JVP own-trajectory constant-LR eight scores"
