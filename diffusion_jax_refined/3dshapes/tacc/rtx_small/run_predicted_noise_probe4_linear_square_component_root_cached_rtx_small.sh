#!/usr/bin/env bash
#SBATCH -J 3d-pn4-lsqroot
#SBATCH -o 3d-pn4-lsqroot-%j.out
#SBATCH -e 3d-pn4-lsqroot-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"

NUM_PROBES=4
QUERY_PATTERN='loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}'
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_probe4_linear_square_component_root/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

run_contraction() {
  local contraction="$1"
  local label="$2"
  local run_id="${SLURM_JOB_ID}_${label}"
  local pids=() failed=0
  echo "[score] ${label}: contraction=${contraction}; probes=P1-P4"
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        python "${DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --epochs "${JAX_EPOCHS}" --run-id "${run_id}" \
          --shard-index "${shard}" --shard-count 2 \
          --num-probes "${NUM_PROBES}" --contraction "${contraction}" \
          --query-namespace-pattern "${QUERY_PATTERN}" \
          --expected-query-probe-mode independent_gaussian
    ) >"${LOG_ROOT}/${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || {
    echo "${label} shard failed; inspect ${LOG_ROOT}" >&2
    exit 1
  }
  python "${DRIVER}" merge \
    --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --epochs "${JAX_EPOCHS}" --run-id "${run_id}" --shard-count 2 \
    --num-probes "${NUM_PROBES}" --contraction "${contraction}" \
    --query-namespace-pattern "${QUERY_PATTERN}" \
    --expected-query-probe-mode independent_gaussian
}

echo "[phase 1/4] linear: mean_r(z_r) per checkpoint/timestamp"
run_contraction signed linear
echo "[phase 2/4] square: mean_r(z_r^2) per checkpoint/timestamp"
run_contraction squared square
echo "[phase 3/4] component root of square: mean_r(sqrt(z_r^2))=mean_r(abs(z_r))"
run_contraction absolute component_root

echo "[phase 4/4] cached LDS for both global orientations"
SCHEMES="predicted_noise_jvp_signed_probe4,predicted_noise_jvp_l2_squared_probe4,predicted_noise_jvp_absolute_probe4"
for sign in 1 -1; do
  JAX_PLATFORMS=cpu python \
    "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
      --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
      --score-schemes "${SCHEMES}" --prediction-sign="${sign}"
done

for sign in p1 m1; do
  python "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
    --experiment "${EXPERIMENT_TAG}" --num-probes 4 \
    --reduction linear --prediction-sign "${sign}"
  python "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
    --experiment "${EXPERIMENT_TAG}" --num-probes 4 \
    --reduction termwise_square --prediction-sign "${sign}"
  python "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
    --experiment "${EXPERIMENT_TAG}" --num-probes 4 \
    --reduction absolute --prediction-sign "${sign}"
done

echo "[done] P1-P4 random directions: linear, square, and component-root scores"
