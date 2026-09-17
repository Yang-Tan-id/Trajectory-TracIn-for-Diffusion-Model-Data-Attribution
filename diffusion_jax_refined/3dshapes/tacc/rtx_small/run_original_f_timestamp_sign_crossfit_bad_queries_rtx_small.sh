#!/usr/bin/env bash
#SBATCH -J 3d-orgf-tscv
#SBATCH -o 3d-orgf-tscv-%j.out
#SBATCH -e 3d-orgf-tscv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_original_f_timestamp_sign_crossfit.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
[[ -f "${SHAPES_ROOT}/script/analyze_original_f_timestamp_sign_crossfit.py" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"

RUN_ID="${STREAM_RUN_ID:-${SLURM_JOB_ID}}"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_timestamp_sign_crossfit_bad_queries/run_${RUN_ID}"
COMPONENT_DIR="${OUT_DIR}/streamed_components"
mkdir -p "${COMPONENT_DIR}"

SOURCE_ARTIFACT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin/train_datapoint_gradient_artifact.npz"
SOURCE_PART_DIR="${SOURCE_ARTIFACT}.parts"
if [[ -d "${SOURCE_PART_DIR}" ]]; then
  source_count="$(find "${SOURCE_PART_DIR}" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
else
  source_count=0
fi
if [[ "${source_count}" != "50" ]]; then
  echo "Expected 50 source projected-gradient parts, found ${source_count}: ${SOURCE_PART_DIR}" >&2
  exit 1
fi
mapfile -t QUERY_ARTIFACTS < <(python "${SHAPES_ROOT}/script/print_query_artifact_paths.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids 1,3,4,8 \
  --namespace loss_direction_residual_rms_original_f)
if [[ "${#QUERY_ARTIFACTS[@]}" != "4" ]]; then
  echo "Expected four original-f query artifacts, found ${#QUERY_ARTIFACTS[@]}" >&2
  exit 1
fi
QUERY_ARTIFACT_LIST="$(IFS=:; echo "${QUERY_ARTIFACTS[*]}")"

export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse
export TRAJ_PARAMETER_SOURCE=raw
export TRAJ_NUM_SNAPSHOTS=10
export TRAJ_TRAIN_MC_SAMPLES=10
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0
export TRAJ_TRACIN_TRAIN_REUSE_GRADIENT_RESIDUAL_RMS=1
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE="${TRAJ_TRACIN_TRAIN_BATCH_DTYPE:-float32}"
export TRAJ_TRACIN_SOURCE_TRAIN_ARTIFACT="${SOURCE_ARTIFACT}"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="${COMPONENT_DIR}/unused_train_artifact.npz"
export TRAJ_TRACIN_RESIDUAL_RMS_STREAM_QUERY_ARTIFACTS="${QUERY_ARTIFACT_LIST}"
export TRAJ_TRACIN_RESIDUAL_RMS_STREAM_OUTPUT="${COMPONENT_DIR}/components"
export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[phase 1/2] stream residual-RMS products directly into timestamp components"
cd "${SHAPES_ROOT}/data_attribution/traj_tracin"
pids=()
for shard in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    export TRAJ_TRACIN_CKPT_SHARD_INDEX="${shard}"
    export TRAJ_TRACIN_CKPT_SHARD_COUNT=2
    python 01_train_datapoint_gradient.py
  ) >"${COMPONENT_DIR}/gpu_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || { echo "Streaming shard failed; inspect ${COMPONENT_DIR}/gpu_*.log" >&2; exit 1; }

echo "[phase 2/2] Q=1,3,4,8; four normalization variants; repeated two-fold crossfit"
CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_original_f_timestamp_sign_crossfit.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids 1,3,4,8 \
  --component-dir "${COMPONENT_DIR}" \
  --repeats 20 \
  --random-seed 20260916 \
  --out-dir "${OUT_DIR}"

echo "[done] original-f timestamp-sign crossfit complete: ${OUT_DIR}"
