#!/usr/bin/env bash
#SBATCH -J 3d-pn7-nop7
#SBATCH -o 3d-pn7-nop7-%j.out
#SBATCH -e 3d-pn7-nop7-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SCORE_SUFFIX="timestamp_shared_leave_p7_out_own_trajectory"
QUERY_PATTERNS="loss_direction_predicted_noise_probe1_timestamp_shared_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed20260918_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed73194261_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed418507293_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed90216487_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed563809241_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed816430927_checkpoint_own_trajectory_r0"
PROBE_SEEDS="20260917,20260918,73194261,418507293,90216487,563809241,816430927"
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/seven_timestamp_shared_probe_leave_p7_out/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

run_contraction() {
  local contraction="$1" label="$2" run_id="${SLURM_JOB_ID}_$2"
  echo "[score] ${label}: 7 probes, P7 excluded"
  local pids=() failed=0
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        python "${SCORE_DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --run-id "${run_id}" --shard-index "${shard}" --shard-count 2 \
          --num-probes 7 --contraction "${contraction}" \
          --namespace-suffix "${SCORE_SUFFIX}" \
          --query-namespace-patterns "${QUERY_PATTERNS}" \
          --expected-query-probe-mode timestamp_shared_gaussian \
          --expected-query-probe-seeds "${PROBE_SEEDS}"
    ) >"${LOG_ROOT}/${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "${label} failed; inspect ${LOG_ROOT}" >&2; exit 1; }
  JAX_PLATFORMS=cpu python "${SCORE_DRIVER}" merge \
    --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --run-id "${run_id}" --shard-count 2 --num-probes 7 \
    --contraction "${contraction}" --namespace-suffix "${SCORE_SUFFIX}" \
    --query-namespace-patterns "${QUERY_PATTERNS}" \
    --expected-query-probe-mode timestamp_shared_gaussian \
    --expected-query-probe-seeds "${PROBE_SEEDS}"
}

run_contraction squared square_mean
run_contraction absolute root_mean
run_contraction probe_l2 term_root
run_contraction timestamp_probe_l2 timestamp_root

SCHEMES="predicted_noise_jvp_l2_squared_probe7_${SCORE_SUFFIX},predicted_noise_jvp_absolute_probe7_${SCORE_SUFFIX},predicted_noise_jvp_probe_l2_probe7_${SCORE_SUFFIX},predicted_noise_jvp_timestamp_probe_l2_probe7_${SCORE_SUFFIX}"
for sign in 1 -1; do
  JAX_PLATFORMS=cpu python "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --score-schemes "${SCHEMES}" --prediction-sign="${sign}"
done

for sign in p1 m1; do
  for reduction in termwise_square absolute probe_l2 timestamp_probe_l2; do
    python "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
      --experiment "${EXPERIMENT_TAG}" --num-probes 7 \
      --namespace-suffix "${SCORE_SUFFIX}" --reduction "${reduction}" \
      --prediction-sign "${sign}"
  done
done

echo "[done] seven-probe leave-P7-out four-combination evaluation"
