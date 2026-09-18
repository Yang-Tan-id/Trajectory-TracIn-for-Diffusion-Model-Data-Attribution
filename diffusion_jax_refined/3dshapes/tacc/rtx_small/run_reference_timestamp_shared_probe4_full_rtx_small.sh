#!/usr/bin/env bash
#SBATCH -J 3d-pn4-ref
#SBATCH -o 3d-pn4-ref-%j.out
#SBATCH -e 3d-pn4-ref-%j.err
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

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
PROBE_SEEDS=(314159265 271828183 161803399 141421357)
PROBE_SEEDS_CSV="314159265,271828183,161803399,141421357"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-reference}"
if [[ "${TRAJECTORY_MODE}" == "reference" ]]; then
  QUERY_NAMESPACE_TEMPLATE='loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed%s_r0'
  INDIVIDUAL_SUFFIX_TEMPLATE='timestamp_shared_reference_seed{seed}'
  COMBINED_SUFFIX="timestamp_shared_reference_probe4_fresh20260917"
elif [[ "${TRAJECTORY_MODE}" == "own" ]]; then
  QUERY_NAMESPACE_TEMPLATE='loss_direction_predicted_noise_probe1_timestamp_shared_own_compare_seed%s_checkpoint_own_trajectory_r0'
  INDIVIDUAL_SUFFIX_TEMPLATE='timestamp_shared_own_compare_seed{seed}'
  COMBINED_SUFFIX="timestamp_shared_own_compare_probe4_fresh20260917"
else
  echo "TRAJECTORY_MODE must be reference or own" >&2
  exit 2
fi
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
LOG_ROOT="${RESULT_ROOT}/logs/${TRAJECTORY_MODE}_timestamp_shared_probe4/${SLURM_JOB_ID}"
OUTDIR="${RESULT_ROOT}/eval/${TRAJECTORY_MODE}_timestamp_shared_probe4/run_${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

if [[ "${TRAJECTORY_MODE}" == "own" ]]; then
  export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
else
  unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY
fi

patterns=()
for seed in "${PROBE_SEEDS[@]}"; do
  printf -v namespace "${QUERY_NAMESPACE_TEMPLATE}" "${seed}"
  patterns+=("${namespace}")
done
QUERY_PATTERNS="$(IFS=,; echo "${patterns[*]}")"

echo "[phase 1/3] four fresh timestamp-shared probes; trajectory_mode=${TRAJECTORY_MODE}"
echo "[invariant] same probe/timestamp direction is shared across every checkpoint and query"
pids=()
for gpu in 0 1; do
  (
    for slot in 0 1; do
      probe=$((gpu * 2 + slot))
      seed="${PROBE_SEEDS[$probe]}"
      namespace="${patterns[$probe]}"
      echo "[gpu ${gpu}] P$((probe + 1)) seed=${seed} namespace=${namespace}"
      python "${QUERY_RUNNER}" \
        --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --epochs "${JAX_EPOCHS}" --query-ids 0,1,2,3,4,5,6,7,8,9 \
        --gpus "${gpu}" --skip-sampling --skip-score \
        --artifact-namespace "${namespace}" \
        --query-objective trajectory_predicted_noise_probe \
        --predicted-noise-probe-index 0 --predicted-noise-probe-count 1 \
        --predicted-noise-probe-mode timestamp_shared_gaussian \
        --predicted-noise-probe-seed "${seed}" \
        --num-snapshots 10 --log-prefix "pn4_ref_seed${seed}_gpu${gpu}"
    done
  ) >"${LOG_ROOT}/query_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "query generation failed; inspect ${LOG_ROOT}" >&2; exit 1; }

score_one_probe() {
  local gpu="$1" probe="$2" contraction="$3" label="$4"
  local seed="${PROBE_SEEDS[$probe]}" pattern="${patterns[$probe]}"
  local suffix="${INDIVIDUAL_SUFFIX_TEMPLATE/\{seed\}/${seed}}"
  local run_id="${SLURM_JOB_ID}_p$((probe + 1))_${label}"
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
}

echo "[phase 2/3] individual linear, square, absolute; two probes per GPU"
pids=()
for gpu in 0 1; do
  (
    for slot in 0 1; do
      probe=$((gpu * 2 + slot))
      score_one_probe "${gpu}" "${probe}" signed linear
      score_one_probe "${gpu}" "${probe}" squared square
      score_one_probe "${gpu}" "${probe}" absolute absolute
    done
  ) >"${LOG_ROOT}/individual_score_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "individual scoring failed; inspect ${LOG_ROOT}" >&2; exit 1; }

score_combined() {
  local contraction="$1"
  local label="$2"
  local run_id="${SLURM_JOB_ID}_combined_${label}"
  local pids=() failed=0
  echo "[combined] ${label}: ${contraction}"
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        python "${SCORE_DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --run-id "${run_id}" --shard-index "${shard}" --shard-count 2 \
          --num-probes 4 --contraction "${contraction}" \
          --namespace-suffix "${COMBINED_SUFFIX}" \
          --query-namespace-patterns "${QUERY_PATTERNS}" \
          --expected-query-probe-mode timestamp_shared_gaussian \
          --expected-query-probe-seeds "${PROBE_SEEDS_CSV}"
    ) >"${LOG_ROOT}/combined_${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "combined ${label} failed; inspect ${LOG_ROOT}" >&2; exit 1; }
  JAX_PLATFORMS=cpu python "${SCORE_DRIVER}" merge \
    --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --run-id "${run_id}" --shard-count 2 --num-probes 4 \
    --contraction "${contraction}" --namespace-suffix "${COMBINED_SUFFIX}" \
    --query-namespace-patterns "${QUERY_PATTERNS}" \
    --expected-query-probe-mode timestamp_shared_gaussian \
    --expected-query-probe-seeds "${PROBE_SEEDS_CSV}"
}

echo "[phase 3/3] four combined reductions"
score_combined squared square_mean
score_combined absolute absolute_mean
score_combined probe_l2 term_root
score_combined timestamp_probe_l2 timestamp_root

JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_reference_timestamp_shared_probe4_scores.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --probe-seeds "${PROBE_SEEDS_CSV}" --combined-suffix "${COMBINED_SUFFIX}" \
  --individual-suffix-template "${INDIVIDUAL_SUFFIX_TEMPLATE}" \
  --out-dir "${OUTDIR}"

echo "[done] ${TRAJECTORY_MODE} four-probe individual and combined scores: ${OUTDIR}"
