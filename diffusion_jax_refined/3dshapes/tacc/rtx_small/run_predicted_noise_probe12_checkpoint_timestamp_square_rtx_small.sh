#!/usr/bin/env bash
#SBATCH -J 3d-pn12-tsq
#SBATCH -o 3d-pn12-tsq-%j.out
#SBATCH -e 3d-pn12-tsq-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"
[[ -f "${SCORE_DRIVER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

NUM_PROBES=12
QUERY_PATTERN='loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}'
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_probe12_checkpoint_timestamp_square/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "[phase 1/2] per checkpoint/probe: mean ten timestamps, square, then mean probes"
pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
        --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --epochs "${JAX_EPOCHS}" --run-id "${SLURM_JOB_ID}" \
        --shard-index "${shard}" --shard-count 2 \
        --num-probes "${NUM_PROBES}" \
        --contraction checkpoint_timestamp_sum_square \
        --query-namespace-pattern "${QUERY_PATTERN}" \
        --expected-query-probe-mode independent_gaussian
  ) >"${LOG_ROOT}/score_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "Score shard failed; inspect ${LOG_ROOT}" >&2; exit 1; }

"${PYTHON_BIN}" "${SCORE_DRIVER}" merge \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --run-id "${SLURM_JOB_ID}" --shard-count 2 \
  --num-probes "${NUM_PROBES}" \
  --contraction checkpoint_timestamp_sum_square \
  --query-namespace-pattern "${QUERY_PATTERN}" \
  --expected-query-probe-mode independent_gaussian

echo "[phase 2/2] cached LDS with energy sign -1"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --score-schemes predicted_noise_jvp_checkpoint_timestamp_sum_square_probe12 \
  --prediction-sign=-1

"${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
  --experiment "${EXPERIMENT_TAG}" --num-probes 12 \
  --reduction checkpoint_timestamp_square --prediction-sign m1

echo "[done] twelve-probe checkpoint timestamp-mean square LDS complete"
