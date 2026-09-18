#!/usr/bin/env bash
#SBATCH -J 3d-rawg-upd
#SBATCH -o 3d-rawg-upd-%j.out
#SBATCH -e 3d-rawg-upd-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_raw_train_gradient_update.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
NAMESPACE="${NAMESPACE:-raw_train_gradient_update_sanity_reference_v2}"
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
TRAIN_ARTIFACT="${RESULT_ROOT}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin/train_datapoint_gradient_artifact.npz"
OUTDIR="${RESULT_ROOT}/eval/raw_train_gradient_update_sanity_reference/run_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_COUNT=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT=1
export TRAJ_TRACIN_PARAMETER_DELTA_JVP_DIAGNOSTIC=1
export TRAJ_TRACIN_RAW_TRAIN_GRADIENT_ARTIFACT="${TRAIN_ARTIFACT}"

python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs 200 --query-ids 0 --gpus 0 \
  --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_predicted_noise_probe \
  --predicted-noise-probe-count 1 --num-snapshots 10 \
  --log-prefix raw_train_gradient_update

JAX_PLATFORMS=cpu python "${SHAPES_ROOT}/script/analyze_raw_train_gradient_update.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query-ids 0 --geometry-namespace "${NAMESPACE}" --out-dir "${OUTDIR}"

echo "[done] raw train-gradient/update sanity: ${OUTDIR}"
