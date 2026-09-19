#!/usr/bin/env bash
#SBATCH -J 3d-adam-scatter
#SBATCH -o 3d-adam-scatter-%j.out
#SBATCH -e 3d-adam-scatter-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_adamw_four_event_original_f_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1

experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
out="$shapes/result/$experiment/eval/adamw_original_f_reference_vs_own_scatter/run_${SLURM_JOB_ID}"
mkdir -p "$out/reference" "$out/own"

for trajectory in reference own; do
  if [[ "$trajectory" == reference ]]; then
    namespace="loss_direction_residual_rms_original_f"
  else
    namespace="loss_direction_original_f_checkpoint_own_trajectory"
  fi
  echo "[plot] $trajectory"
  JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
    --experiment "$experiment" \
    --train-seed "$train_seed" \
    --query-namespace "$namespace" \
    --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
    --checkpoint-weighting uniform \
    --plot-selection \
    --out-dir "$out/$trajectory"
done

tar -C "$out" -czf "$out/lds_scatter_reference_vs_own.tar.gz" \
  reference/lds_scatter_four_residual_query_train_l2_endpoint \
  own/lds_scatter_four_residual_query_train_l2_endpoint
echo "[saved] $out/lds_scatter_reference_vs_own.tar.gz"
