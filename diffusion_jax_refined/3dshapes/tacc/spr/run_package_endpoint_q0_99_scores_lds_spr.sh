#!/usr/bin/env bash
#SBATCH -J pack-3d-q100
#SBATCH -p spr
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -t 04:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
marker="diffusion_jax_refined/3dshapes/script/package_endpoint_q0_99_scores_lds.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do repo="$(dirname "$repo")"; done
[[ -f "$repo/$marker" ]] || { echo "cannot locate repository; set REPO_ROOT" >&2; exit 1; }

python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
stamp="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-${SCRATCH}/shape3d_q0_99_endpoint_export_${stamp}}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "[definition] existing Shape3D Q0-Q99 endpoint score banks and cached LDS only"
echo "[families] endpoint-pollute 10/100; endpoint-TracIn; ReTrac; raw and AdamW"
echo "[output] $output_dir"

"$python_bin" "$repo/$marker" \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-ids "${QUERY_IDS:-$(seq -s, 0 99)}" \
  --output-dir "$output_dir"

echo "[done] $output_dir"
