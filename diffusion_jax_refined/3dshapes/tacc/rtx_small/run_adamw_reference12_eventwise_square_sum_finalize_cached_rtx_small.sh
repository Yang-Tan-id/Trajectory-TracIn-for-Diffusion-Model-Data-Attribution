#!/usr/bin/env bash
#SBATCH -J 3d-a12-evfin
#SBATCH -o 3d-a12-evfin-%j.out
#SBATCH -e 3d-a12-evfin-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00

set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/combine_eventwise_squared_scores.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
exp="${EXPERIMENT_TAG:-experiment1}"; seed="${TRAIN_SEED:-42}"
raw_inputs='predicted_noise_jvp_l2_squared_probe12_adamw4_e1_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e2_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e3_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e4_reference12'
res_inputs='predicted_noise_jvp_l2_squared_probe12_adamw4_e1_residual_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e2_residual_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e3_residual_reference12,predicted_noise_jvp_l2_squared_probe12_adamw4_e4_residual_reference12'
raw_output='predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_square_sum_reference12'
res_output='predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_residual_square_sum_reference12'
python "$shapes/script/combine_eventwise_squared_scores.py" --experiment "$exp" --train-seed "$seed" --input-namespaces "$raw_inputs" --output-namespace "$raw_output"
python "$shapes/script/combine_eventwise_squared_scores.py" --experiment "$exp" --train-seed "$seed" --input-namespaces "$res_inputs" --output-namespace "$res_output"
JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" --execute --experiment "$exp" --train-seed "$seed" --score-schemes "$raw_output,$res_output" --prediction-sign=1
echo "[done] finalized cached reference12 eventwise-square LDS"
