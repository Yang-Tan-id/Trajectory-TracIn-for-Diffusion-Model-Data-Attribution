#!/usr/bin/env bash
#SBATCH -J 3d-pos-hemi
#SBATCH -o 3d-pos-hemi-%j.out
#SBATCH -e 3d-pos-hemi-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_positive_probe_timestamp_hemisphere.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SOURCE_INDIVIDUAL_RUN_ID="${SOURCE_INDIVIDUAL_RUN_ID:-3511956}"
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
INPUT="${RESULT_ROOT}/eval/eight_timestamp_shared_probes_individual/run_${SOURCE_INDIVIDUAL_RUN_ID}/per_query.csv"
OUTDIR="${RESULT_ROOT}/eval/positive_probe_timestamp_hemisphere/run_${SLURM_JOB_ID}"

python "${SHAPES_ROOT}/script/analyze_positive_probe_timestamp_hemisphere.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --individual-results "${INPUT}" \
  --reduction root \
  --sign p1 \
  --target endpoint_contarfactual \
  --out-dir "${OUTDIR}"

echo "[done] positive-probe timestamp hemisphere: ${OUTDIR}"
