#!/usr/bin/env bash
#SBATCH -J 3d-v-delta
#SBATCH -o 3d-v-delta-%j.out
#SBATCH -e 3d-v-delta-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

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
GEOMETRY_NAMESPACE="${GEOMETRY_NAMESPACE:-reference_probe_delta_geometry_collect_all}"
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
OUTDIR="${RESULT_ROOT}/eval/reference_probe_delta_alignment/run_${SLURM_JOB_ID}"

# The individual score artifacts do not contain raw delta-epsilon tensors.
# Materialize them once with forward passes only.  DELTA_CONTINUITY turns on the
# existing compact collect-all payload; the independent alignment probe itself
# is irrelevant because the analyzer reconstructs the exact 12 timestamp-shared
# probes from their seeds.
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY
export TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_COUNT=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_DELTA_CONTINUITY=1

echo "[phase 1/2] fixed-reference checkpoint delta-epsilon outputs (forward only)"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs 200 --query-ids 0,1,2,3,4,5,6,7,8,9 --gpus 0,1 \
  --skip-sampling --skip-score \
  --artifact-namespace "${GEOMETRY_NAMESPACE}" \
  --query-objective trajectory_predicted_noise_probe \
  --predicted-noise-probe-count 1 --num-snapshots 10 \
  --log-prefix reference_v_delta_geometry

unset TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY
unset TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT
unset TRAJ_TRACIN_PROBE_ALIGNMENT_DELTA_CONTINUITY

echo "[phase 2/2] reconstruct the 12 timestamp-shared probes and compare LDS groups"
python "${SHAPES_ROOT}/script/analyze_reference_probe_delta_alignment.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --geometry-namespace "${GEOMETRY_NAMESPACE}" \
  --lds-csv "${RESULT_ROOT}/eval/reference_timestamp_shared_probe4/run_${FIRST4_RUN_ID}/per_query.csv" \
  --lds-csv "${RESULT_ROOT}/eval/reference_timestamp_shared_probe8/run_${FRESH8_RUN_ID}/per_query.csv" \
  --threshold 5 --out-dir "${OUTDIR}"

echo "[done] fixed-reference probe/delta alignment: ${OUTDIR}"
