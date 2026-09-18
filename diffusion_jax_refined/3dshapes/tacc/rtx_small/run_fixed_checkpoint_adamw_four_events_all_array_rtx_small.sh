#!/usr/bin/env bash
#SBATCH -J 3d-fixadam4
#SBATCH -o 3d-fixadam4-%A_%a.out
#SBATCH -e 3d-fixadam4-%A_%a.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH --array=0-48%2
#SBATCH -t 24:00:00

set -euo pipefail
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/replay_exact_training_interval.py" ]]; do
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
PROJ_DIM="${PROJ_DIM:-4096}"
START_EPOCH=$((4 * (SLURM_ARRAY_TASK_ID + 1)))
END_EPOCH=$((START_EPOCH + 4))
OUTDIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/fixed_checkpoint_adamw_four_events/epoch_${START_EPOCH}_${END_EPOCH}"
LOGDIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/fixed_checkpoint_adamw_four_events/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}" "${LOGDIR}"

for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="${gpu}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
    python "${SHAPES_ROOT}/script/replay_exact_training_interval.py" \
      --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
      --start-epoch "${START_EPOCH}" --end-epoch "${END_EPOCH}" \
      --fixed-checkpoint --extract-gradient-sketches \
      --event-feature adamw_hypothetical_update --proj-dim "${PROJ_DIM}" \
      --shard-id "${gpu}" --num-shards 2 --out-dir "${OUTDIR}" \
      >"${LOGDIR}/gpu_${gpu}.log" 2>&1 &
done
wait
echo "[done] fixed-checkpoint AdamW four-event features: ${OUTDIR}"
