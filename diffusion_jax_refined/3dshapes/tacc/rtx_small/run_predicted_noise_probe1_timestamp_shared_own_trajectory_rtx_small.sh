#!/usr/bin/env bash
#SBATCH -J 3d-pn1-tsown
#SBATCH -o 3d-pn1-tsown-%j.out
#SBATCH -e 3d-pn1-tsown-%j.err
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
QUERY_RUNNER="${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PROBE_SEED="${PROBE_SEED:-20260917}"
export PROBE_TAG="${PROBE_TAG:-}"

if [[ -n "${PROBE_TAG}" && ! "${PROBE_TAG}" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "PROBE_TAG must contain only letters, digits, and underscores" >&2
  exit 2
fi

NUM_PROBES=1
TAG_SUFFIX="${PROBE_TAG:+_${PROBE_TAG}}"
NAMESPACE_BASE="loss_direction_predicted_noise_probe1_timestamp_shared${TAG_SUFFIX}_checkpoint_own_trajectory"
QUERY_PATTERN="${NAMESPACE_BASE}_r{probe_index}"
SCORE_SUFFIX="timestamp_shared${TAG_SUFFIX}_own_trajectory"
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_probe1_timestamp_shared${TAG_SUFFIX}_own_trajectory/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "[phase 1/3] one Gaussian probe per timestamp, shared over all checkpoints and queries"
  export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
  pids=()
  for gpu in 0 1; do
    if [[ "${gpu}" == "0" ]]; then
      query_ids="0,1,2,3,4"
    else
      query_ids="5,6,7,8,9"
    fi
    (
      python "${QUERY_RUNNER}" \
        --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --epochs "${JAX_EPOCHS}" --query-ids "${query_ids}" \
        --gpus "${gpu}" --skip-sampling --skip-score \
        --artifact-namespace "${NAMESPACE_BASE}_r0" \
        --query-objective trajectory_predicted_noise_probe \
        --predicted-noise-probe-index 0 \
        --predicted-noise-probe-count "${NUM_PROBES}" \
        --predicted-noise-probe-mode timestamp_shared_gaussian \
        --predicted-noise-probe-seed "${PROBE_SEED}" \
        --num-snapshots 10 --log-prefix "pn1_timestamp_shared_own_gpu${gpu}"
    ) >"${LOG_ROOT}/query_gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || {
    echo "query probe generation failed; inspect ${LOG_ROOT}" >&2
    exit 1
  }
  unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY
else
  echo "[reuse] EVAL_ONLY=1; reusing completed query and score artifacts"
fi

run_contraction() {
  local contraction="$1"
  local label="$2"
  local run_id="${SLURM_JOB_ID}_${label}"
  echo "[score] ${label}: contraction=${contraction}"
  local pids=() failed=0
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        python "${SCORE_DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --epochs "${JAX_EPOCHS}" --run-id "${run_id}" \
          --shard-index "${shard}" --shard-count 2 \
          --num-probes "${NUM_PROBES}" --contraction "${contraction}" \
          --namespace-suffix "${SCORE_SUFFIX}" \
          --query-namespace-pattern "${QUERY_PATTERN}" \
          --expected-query-probe-mode timestamp_shared_gaussian \
          --expected-query-probe-seed "${PROBE_SEED}"
    ) >"${LOG_ROOT}/${label}_score_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || {
    echo "${label} score generation failed; inspect ${LOG_ROOT}" >&2
    exit 1
  }
  python "${SCORE_DRIVER}" merge \
    --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --epochs "${JAX_EPOCHS}" --run-id "${run_id}" --shard-count 2 \
    --num-probes "${NUM_PROBES}" --contraction "${contraction}" \
    --namespace-suffix "${SCORE_SUFFIX}" \
    --query-namespace-pattern "${QUERY_PATTERN}" \
    --expected-query-probe-mode timestamp_shared_gaussian \
    --expected-query-probe-seed "${PROBE_SEED}"
}

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "[phase 2/3] linear, square, and component-root scores"
  run_contraction signed linear
  run_contraction squared square
  run_contraction probe_l2 component_root
fi

echo "[phase 3/3] LDS for p1 and m1"
SCHEMES="predicted_noise_jvp_signed_${SCORE_SUFFIX},predicted_noise_jvp_l2_squared_${SCORE_SUFFIX},predicted_noise_jvp_probe_l2_${SCORE_SUFFIX}"
for sign in 1 -1; do
  JAX_PLATFORMS=cpu python "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --score-schemes "${SCHEMES}" --prediction-sign="${sign}"
done

for sign in p1 m1; do
  for reduction in linear termwise_square probe_l2; do
    python "${SHAPES_ROOT}/script/print_predicted_noise_timestamp_checkpoint_square_lds.py" \
      --experiment "${EXPERIMENT_TAG}" --num-probes 1 \
      --namespace-suffix "${SCORE_SUFFIX}" \
      --reduction "${reduction}" --prediction-sign "${sign}"
  done
done

echo "[done] one timestamp-shared probe on checkpoint-own trajectories"
