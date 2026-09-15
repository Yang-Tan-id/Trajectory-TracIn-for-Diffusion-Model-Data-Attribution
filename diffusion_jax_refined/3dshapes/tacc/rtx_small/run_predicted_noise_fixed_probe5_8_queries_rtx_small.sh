#!/usr/bin/env bash
#SBATCH -J 3d-pnfix5-8q
#SBATCH -o 3d-pnfix5-8q-%j.out
#SBATCH -e 3d-pnfix5-8q-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do candidate="$(dirname "${candidate}")"; done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
QUERY_DRIVER="${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py"
[[ -f "${QUERY_DRIVER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[phase 1/1] extend fixed orthogonal bank with probes 5-8"
for probe_index in 4 5 6 7; do
  namespace="predicted_noise_shared_orthogonal_probe4_r${probe_index}"
  "${PYTHON_BIN}" "${QUERY_DRIVER}" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --epochs "${JAX_EPOCHS}" --query-ids 0,1,2,3,4,5,6,7,8,9 --gpus 0,1 \
    --skip-sampling --skip-score --artifact-namespace "${namespace}" \
    --query-objective trajectory_predicted_noise_probe \
    --predicted-noise-probe-index "${probe_index}" \
    --predicted-noise-probe-mode shared_orthogonal_extended \
    --predicted-noise-probe-count 8 \
    --num-snapshots 10 --log-prefix "orth_probe_${probe_index}" --python-bin "${PYTHON_BIN}"
done
echo "[done] fixed orthogonal probes 5-8 are complete"
