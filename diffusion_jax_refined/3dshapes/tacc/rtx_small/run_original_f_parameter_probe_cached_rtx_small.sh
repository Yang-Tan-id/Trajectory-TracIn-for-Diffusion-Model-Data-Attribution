#!/usr/bin/env bash
#SBATCH -J 3d-orgf-pv
#SBATCH -o 3d-orgf-pv-%j.out
#SBATCH -e 3d-orgf-pv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_original_f_parameter_probe.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export NUM_PROBES="${NUM_PROBES:-1}"
export PROBE_SEED="${PROBE_SEED:-20260917}"
export ORIGINAL_QUERY_NAMESPACE="${ORIGINAL_QUERY_NAMESPACE:-loss_direction_original_f_checkpoint_own_trajectory}"

OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_parameter_probe${NUM_PROBES}/run_${SLURM_JOB_ID}"

JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_original_f_parameter_probe.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --num-probes "${NUM_PROBES}" \
  --probe-seed "${PROBE_SEED}" \
  --original-query-namespace "${ORIGINAL_QUERY_NAMESPACE}" \
  --repeats 20 \
  --random-seed 20260916 \
  --out-dir "${OUT_DIR}"

echo "[done] original-f parameter probe: ${OUT_DIR}"
