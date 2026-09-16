#!/usr/bin/env bash
#SBATCH -J 3d-pn12-angle
#SBATCH -o 3d-pn12-angle-%j.out
#SBATCH -e 3d-pn12-angle-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
resolve_repo_root() {
  local start candidate
  for start in "${REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}"; do
    [[ -n "${start}" && -d "${start}" ]] || continue
    candidate="$(cd "${start}" && pwd)"
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || {
  echo "Could not locate repository from REPO_ROOT, SLURM_SUBMIT_DIR, or SCRIPT_DIR" >&2
  exit 1
}
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_COUNT=12

echo "[phase 1/2] forward-only predicted-noise/probe angles for all queries"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 \
  --skip-sampling \
  --skip-score \
  --artifact-namespace predicted_noise_alignment_probe12 \
  --query-objective trajectory_predicted_noise_probe \
  --predicted-noise-probe-count 12 \
  --num-snapshots 10 \
  --log-prefix angle12

echo "[phase 2/2] orient saved projected query gradients and evaluate LDS"
unset TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_predicted_noise_angle_oriented_scores.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --num-probes 12 \
  --run-id "${SLURM_JOB_ID}" \
  --prediction-sign 1

echo "[done] angle-oriented predicted-noise analysis complete"
