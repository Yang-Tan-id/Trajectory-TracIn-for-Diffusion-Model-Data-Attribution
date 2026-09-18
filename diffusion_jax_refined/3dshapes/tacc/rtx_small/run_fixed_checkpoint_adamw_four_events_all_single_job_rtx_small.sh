#!/usr/bin/env bash
#SBATCH -J 3d-fixadam-all
#SBATCH -o 3d-fixadam-all-%j.out
#SBATCH -e 3d-fixadam-all-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

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
ATTRIBUTION_POINTS="${ATTRIBUTION_POINTS:-5000}"
FIRST_INTERVAL="${FIRST_INTERVAL:-0}"
LAST_INTERVAL="${LAST_INTERVAL:-48}"
LOGROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/fixed_checkpoint_adamw_four_events_n${ATTRIBUTION_POINTS}/${SLURM_JOB_ID}"
mkdir -p "${LOGROOT}"

for interval in $(seq "${FIRST_INTERVAL}" "${LAST_INTERVAL}"); do
  start_epoch=$((4 * (interval + 1)))
  end_epoch=$((start_epoch + 4))
  outdir="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/fixed_checkpoint_adamw_four_events_n${ATTRIBUTION_POINTS}/epoch_${start_epoch}_${end_epoch}"
  logdir="${LOGROOT}/interval_${interval}_epoch_${start_epoch}_${end_epoch}"
  mkdir -p "${outdir}" "${logdir}"
  echo "[interval $((interval + 1))/49] checkpoint epoch ${start_epoch} -> ${end_epoch}"

  for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="${gpu}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "${SHAPES_ROOT}/script/replay_exact_training_interval.py" \
        --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --start-epoch "${start_epoch}" --end-epoch "${end_epoch}" \
        --fixed-checkpoint --extract-gradient-sketches \
        --event-feature adamw_hypothetical_update --proj-dim "${PROJ_DIM}" \
        --attribution-points "${ATTRIBUTION_POINTS}" \
        --shard-id "${gpu}" --num-shards 2 --out-dir "${outdir}" \
        >"${logdir}/gpu_${gpu}.log" 2>&1 &
  done
  wait
  echo "[interval complete] ${start_epoch}->${end_epoch}: ${outdir}"
done

echo "[done] all 49 fixed-checkpoint AdamW four-event intervals"
