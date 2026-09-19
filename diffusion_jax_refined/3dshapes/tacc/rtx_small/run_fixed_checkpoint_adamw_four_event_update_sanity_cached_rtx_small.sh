#!/usr/bin/env bash
#SBATCH -J 3d-fixadam-san
#SBATCH -o 3d-fixadam-san-%j.out
#SBATCH -e 3d-fixadam-san-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00

set -euo pipefail
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_fixed_checkpoint_adamw_four_event_update.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
ATTRIBUTION_POINTS="${ATTRIBUTION_POINTS:-5000}"
OUTDIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/fixed_checkpoint_adamw_four_event_update_sanity/run_${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
JAX_PLATFORMS=cpu python "${SHAPES_ROOT}/script/analyze_fixed_checkpoint_adamw_four_event_update.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --attribution-points "${ATTRIBUTION_POINTS}" \
  --out-dir "${OUTDIR}"

echo "[done] fixed-checkpoint AdamW four-event update sanity: ${OUTDIR}"
