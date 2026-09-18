#!/usr/bin/env bash
#SBATCH -J 3d-ref12-ctsq
#SBATCH -o 3d-ref12-ctsq-%j.out
#SBATCH -e 3d-ref12-ctsq-%j.err
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
NAMESPACE_SUFFIX=reference_timestamp_shared12
SEEDS=(
  314159265 271828183 161803399 141421357
  538102947 794615203 126937481 682450719
  905173624 417286953 263590817 849031576
)
SEEDS_CSV="$(IFS=,; echo "${SEEDS[*]}")"
patterns=()
for seed in "${SEEDS[@]}"; do
  patterns+=("loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${seed}_r0")
done
PATTERNS_CSV="$(IFS=,; echo "${patterns[*]}")"

LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/reference_probe12_checkpoint_trajectory_square/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

echo "[score] fixed reference: within each checkpoint/probe, mean signed linear terms over trajectory timestamps, then square"
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
        --query-namespace-patterns "${PATTERNS_CSV}" \
        --expected-query-probe-mode timestamp_shared_gaussian \
        --expected-query-probe-seeds "${SEEDS_CSV}" \
        --namespace-suffix "${NAMESPACE_SUFFIX}"
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
  --query-namespace-patterns "${PATTERNS_CSV}" \
  --expected-query-probe-mode timestamp_shared_gaussian \
  --expected-query-probe-seeds "${SEEDS_CSV}" \
  --namespace-suffix "${NAMESPACE_SUFFIX}"

SCORE_SCHEME=predicted_noise_jvp_checkpoint_timestamp_sum_square_probe12_reference_timestamp_shared12
echo "[LDS] evaluate both fixed global orientations; no oracle selection"
for sign in 1 -1; do
  JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --score-schemes "${SCORE_SCHEME}" --prediction-sign="${sign}"
done

for sign in p1 m1; do
  "${PYTHON_BIN}" "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
    --experiment "${EXPERIMENT_TAG}" --num-probes "${NUM_PROBES}" \
    --reduction checkpoint_timestamp_square --prediction-sign "${sign}" \
    --namespace-suffix "${NAMESPACE_SUFFIX}"
done

echo "[done] fixed-reference 12-probe checkpoint/trajectory-linear-sum square LDS"
