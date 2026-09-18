#!/usr/bin/env bash
#SBATCH -J 3d-v-delta
#SBATCH -o 3d-v-delta-%j.out
#SBATCH -e 3d-v-delta-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_reference_probe_delta_alignment.py" ]]; do
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
FIRST4_RUN_ID="${FIRST4_RUN_ID:-3512133}"
FRESH8_RUN_ID="${FRESH8_RUN_ID:-3512746}"
GEOMETRY_NAMESPACE="${GEOMETRY_NAMESPACE:-predicted_noise_output_next_original12}"
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
OUTDIR="${RESULT_ROOT}/eval/reference_probe_delta_alignment/run_${SLURM_JOB_ID}"

python "${SHAPES_ROOT}/script/analyze_reference_probe_delta_alignment.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --geometry-namespace "${GEOMETRY_NAMESPACE}" \
  --lds-csv "${RESULT_ROOT}/eval/reference_timestamp_shared_probe4/run_${FIRST4_RUN_ID}/per_query.csv" \
  --lds-csv "${RESULT_ROOT}/eval/reference_timestamp_shared_probe8/run_${FRESH8_RUN_ID}/per_query.csv" \
  --threshold 5 --out-dir "${OUTDIR}"

echo "[done] fixed-reference probe/delta alignment: ${OUTDIR}"
