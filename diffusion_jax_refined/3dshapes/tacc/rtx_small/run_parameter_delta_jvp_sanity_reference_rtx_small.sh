#!/usr/bin/env bash
#SBATCH -J 3d-param-jvp
#SBATCH -o 3d-param-jvp-%j.out
#SBATCH -e 3d-param-jvp-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail

candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_parameter_delta_jvp_sanity.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
QUERY_IDS="${QUERY_IDS:-0,3,7,8}"
NAMESPACE="${NAMESPACE:-parameter_delta_jvp_sanity_reference}"
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
LOG_ROOT="${RESULT_ROOT}/logs/parameter_delta_jvp_sanity_reference/${SLURM_JOB_ID}"
OUTDIR="${RESULT_ROOT}/eval/parameter_delta_jvp_sanity_reference/run_${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY
export TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_COUNT=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT=1
export TRAJ_TRACIN_PARAMETER_DELTA_JVP_DIAGNOSTIC=1

echo "[phase 1/2] exact J(params[c+1]-params[c]) on fixed reference trajectories; queries=${QUERY_IDS}"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs 200 --query-ids "${QUERY_IDS}" --gpus 0,1 \
  --skip-sampling --skip-score \
  --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_predicted_noise_probe \
  --predicted-noise-probe-count 1 --num-snapshots 10 \
  --log-prefix parameter_delta_jvp_sanity

unset TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY
unset TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT
unset TRAJ_TRACIN_PARAMETER_DELTA_JVP_DIAGNOSTIC

echo "[phase 2/2] summarize 49 checkpoint pairs x 10 timestamps"
JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_parameter_delta_jvp_sanity.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query-ids "${QUERY_IDS}" --geometry-namespace "${NAMESPACE}" \
  --out-dir "${OUTDIR}"

echo "[done] exact parameter-delta JVP sanity: ${OUTDIR}"
