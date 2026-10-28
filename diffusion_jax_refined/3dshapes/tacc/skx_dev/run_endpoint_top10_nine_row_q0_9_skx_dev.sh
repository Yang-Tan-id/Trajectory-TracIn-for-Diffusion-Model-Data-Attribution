#!/usr/bin/env bash
#SBATCH -J 3d-nine-row
#SBATCH -p skx-dev
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00

set -euo pipefail

repo="${REPO_ROOT:-$WORK/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
py="${PYTHON_BIN:-$SCRATCH/conda-envs/trajectory-tracin/bin/python}"
output_dir="${OUTPUT_DIR:-$repo/diffusion_jax_refined/3dshapes/result/experiment1/eval/endpoint_top10_nine_row_q0_9}"

cd "$repo"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/matplotlib-${SLURM_JOB_ID:-manual}}"
mkdir -p "$MPLCONFIGDIR" "$output_dir"

echo "[definition] Shape3D Q0-Q9 endpoint top-10; nine rows per transform"
echo "[direction] PINT timestamp-sum-squared: descending for raw and AdamW"
echo "[direction] TracIn/ReTrac linear raw-gradient: descending"
echo "[direction] TracIn/ReTrac linear AdamW-full: ascending"
echo "[output] $output_dir"

"$py" -m unittest \
  diffusion_jax_refined.tests.test_shape3d_endpoint_nine_row_directions

"$py" \
  diffusion_jax_refined/3dshapes/script/plot_endpoint_nine_row_top10.py \
  --query-ids "$(seq -s, 0 9)" \
  --transforms raw_gradient,adamw_full \
  --top-k 10 \
  --output-dir "$output_dir"

echo "[done] $output_dir"
