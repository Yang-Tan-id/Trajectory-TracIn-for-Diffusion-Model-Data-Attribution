#!/usr/bin/env bash
#SBATCH -J 3d-pn8-ind
#SBATCH -o 3d-pn8-ind-%j.out
#SBATCH -e 3d-pn8-ind-%j.err
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
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SEEDS=(20260917 20260918 73194261 418507293 90216487 563809241 247196803 816430927)
PATTERNS=(
  loss_direction_predicted_noise_probe1_timestamp_shared_checkpoint_own_trajectory_r0
  loss_direction_predicted_noise_probe1_timestamp_shared_seed20260918_checkpoint_own_trajectory_r0
  loss_direction_predicted_noise_probe1_timestamp_shared_seed73194261_checkpoint_own_trajectory_r0
  loss_direction_predicted_noise_probe1_timestamp_shared_seed418507293_checkpoint_own_trajectory_r0
  loss_direction_predicted_noise_probe1_timestamp_shared_seed90216487_checkpoint_own_trajectory_r0
  loss_direction_predicted_noise_probe1_timestamp_shared_seed563809241_checkpoint_own_trajectory_r0
  loss_direction_predicted_noise_probe1_timestamp_shared_seed247196803_checkpoint_own_trajectory_r0
  loss_direction_predicted_noise_probe1_timestamp_shared_seed816430927_checkpoint_own_trajectory_r0
)
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/eight_timestamp_shared_probes_individual/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

score_probe() {
  local gpu="$1" index="$2" seed="${SEEDS[$2]}" pattern="${PATTERNS[$2]}"
  local suffix="timestamp_shared_individual_seed${seed}_own_trajectory"
  for spec in "squared:square" "probe_l2:root" "timestamp_probe_l2:timestamp_root"; do
    local contraction="${spec%%:*}" label="${spec##*:}"
    local run_id="${SLURM_JOB_ID}_p$((index + 1))_${label}"
    echo "[gpu ${gpu}] probe=$((index + 1))/8 seed=${seed} reduction=${label}"
    CUDA_VISIBLE_DEVICES="${gpu}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "${SCORE_DRIVER}" score-shard \
        --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --run-id "${run_id}" --shard-index 0 --shard-count 1 \
        --num-probes 1 --contraction "${contraction}" \
        --namespace-suffix "${suffix}" --query-namespace-pattern "${pattern}" \
        --expected-query-probe-mode timestamp_shared_gaussian \
        --expected-query-probe-seed "${seed}"
    JAX_PLATFORMS=cpu python "${SCORE_DRIVER}" merge \
      --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
      --run-id "${run_id}" --shard-count 1 --num-probes 1 \
      --contraction "${contraction}" --namespace-suffix "${suffix}" \
      --query-namespace-pattern "${pattern}" \
      --expected-query-probe-mode timestamp_shared_gaussian \
      --expected-query-probe-seed "${seed}"
  done
}

pids=()
for gpu in 0 1; do
  (
    for slot in 0 1 2 3; do
      score_probe "${gpu}" $((gpu * 4 + slot))
    done
  ) >"${LOG_ROOT}/score_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "individual scoring failed; inspect ${LOG_ROOT}" >&2; exit 1; }

SEED_CSV="$(IFS=,; echo "${SEEDS[*]}")"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/eight_timestamp_shared_probes_individual/run_${SLURM_JOB_ID}"
JAX_PLATFORMS=cpu python "${SHAPES_ROOT}/script/analyze_individual_timestamp_shared_probes.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --probe-seeds "${SEED_CSV}" --out-dir "${OUT_DIR}"

echo "[done] individual LDS for all eight timestamp-shared probes: ${OUT_DIR}"
