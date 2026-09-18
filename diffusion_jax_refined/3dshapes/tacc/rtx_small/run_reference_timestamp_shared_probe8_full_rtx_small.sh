#!/usr/bin/env bash
#SBATCH -J 3d-pn8-ref
#SBATCH -o 3d-pn8-ref-%j.out
#SBATCH -e 3d-pn8-ref-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

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
PROBE_SET_TAG="${PROBE_SET_TAG:-fresh20260917b}"
PROBE_SEEDS_CSV="${PROBE_SEEDS_CSV:-604938271,857302149,193746825,468025713,725914603,350681927,981374205,246805319}"
IFS=',' read -r -a PROBE_SEEDS <<<"${PROBE_SEEDS_CSV}"
if [[ "${#PROBE_SEEDS[@]}" -ne 8 ]]; then
  echo "PROBE_SEEDS_CSV must contain exactly 8 comma-separated seeds" >&2
  exit 2
fi
if [[ ! "${PROBE_SET_TAG}" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "PROBE_SET_TAG must contain only letters, digits, and underscores" >&2
  exit 2
fi
declare -A seen_seeds=()
for seed in "${PROBE_SEEDS[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]] || [[ -n "${seen_seeds[$seed]:-}" ]]; then
    echo "probe seeds must be distinct nonnegative integers: ${PROBE_SEEDS_CSV}" >&2
    exit 2
  fi
  seen_seeds["${seed}"]=1
done
COMBINED_SUFFIX="timestamp_shared_reference_probe8_${PROBE_SET_TAG}"
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
LOG_ROOT="${RESULT_ROOT}/logs/reference_timestamp_shared_probe8/${SLURM_JOB_ID}"
OUTDIR="${RESULT_ROOT}/eval/reference_timestamp_shared_probe8/run_${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

# Force the common final/reference trajectory even under sbatch --export=ALL.
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

patterns=()
for seed in "${PROBE_SEEDS[@]}"; do
  patterns+=("loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${seed}_r0")
done
QUERY_PATTERNS="$(IFS=,; echo "${patterns[*]}")"

echo "[phase 1/3] eight fresh timestamp-shared probes on one fixed reference trajectory"
echo "[invariant] same probe/timestamp direction is shared across every checkpoint and query"
pids=()
for gpu in 0 1; do
  (
    for slot in 0 1 2 3; do
      probe=$((gpu * 4 + slot))
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
        --num-snapshots 10 --log-prefix "pn8_ref_seed${seed}_gpu${gpu}"
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
  local suffix="timestamp_shared_reference_seed${seed}"
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

echo "[phase 2/3] individual linear, square, absolute; four probes per GPU"
pids=()
for gpu in 0 1; do
  (
    for slot in 0 1 2 3; do
      probe=$((gpu * 4 + slot))
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
          --num-probes 8 --contraction "${contraction}" \
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
    --run-id "${run_id}" --shard-count 2 --num-probes 8 \
    --contraction "${contraction}" --namespace-suffix "${COMBINED_SUFFIX}" \
    --query-namespace-patterns "${QUERY_PATTERNS}" \
    --expected-query-probe-mode timestamp_shared_gaussian \
    --expected-query-probe-seeds "${PROBE_SEEDS_CSV}"
}

echo "[phase 3/3] four eight-probe combined reductions"
score_combined squared square_mean
score_combined absolute absolute_mean
score_combined probe_l2 term_root
score_combined timestamp_probe_l2 timestamp_root

JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_reference_timestamp_shared_probe4_scores.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --probe-seeds "${PROBE_SEEDS_CSV}" --combined-suffix "${COMBINED_SUFFIX}" \
  --out-dir "${OUTDIR}"

echo "[done] fixed-reference eight-probe individual and combined scores: ${OUTDIR}"
