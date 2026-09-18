#!/usr/bin/env bash
#SBATCH -J 3d-pn12-ref
#SBATCH -o 3d-pn12-ref-%j.out
#SBATCH -e 3d-pn12-ref-%j.err
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

# The original fixed-reference P1-P4 plus the later eight-probe set.
PROBE_SEEDS=(
  314159265 271828183 161803399 141421357
  538102947 794615203 126937481 682450719
  905173624 417286953 263590817 849031576
)
PROBE_SEEDS_CSV="$(IFS=,; echo "${PROBE_SEEDS[*]}")"
COMBINED_SUFFIX="timestamp_shared_reference_probe12_first4_plus_fresh8"
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
LOG_ROOT="${RESULT_ROOT}/logs/reference_timestamp_shared_probe12/${SLURM_JOB_ID}"
OUTDIR="${RESULT_ROOT}/eval/reference_timestamp_shared_probe12/run_${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

patterns=()
for seed in "${PROBE_SEEDS[@]}"; do
  patterns+=("loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${seed}_r0")
done
QUERY_PATTERNS="$(IFS=,; echo "${patterns[*]}")"

score_combined() {
  local contraction="$1"
  local label="$2"
  local run_id="${SLURM_JOB_ID}_combined_${label}"
  local pids=() failed=0
  echo "[combined] ${label}: ${contraction}; cached 4+8=12 probes"
  for shard in 0 1; do
    (
      CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
        python "${SCORE_DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
          --run-id "${run_id}" --shard-index "${shard}" --shard-count 2 \
          --num-probes 12 --contraction "${contraction}" \
          --namespace-suffix "${COMBINED_SUFFIX}" \
          --query-namespace-patterns "${QUERY_PATTERNS}" \
          --expected-query-probe-mode timestamp_shared_gaussian \
          --expected-query-probe-seeds "${PROBE_SEEDS_CSV}"
    ) >"${LOG_ROOT}/${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "${label} failed; inspect ${LOG_ROOT}" >&2; exit 1; }
  JAX_PLATFORMS=cpu python "${SCORE_DRIVER}" merge \
    --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --run-id "${run_id}" --shard-count 2 --num-probes 12 \
    --contraction "${contraction}" --namespace-suffix "${COMBINED_SUFFIX}" \
    --query-namespace-patterns "${QUERY_PATTERNS}" \
    --expected-query-probe-mode timestamp_shared_gaussian \
    --expected-query-probe-seeds "${PROBE_SEEDS_CSV}"
}

echo "[cached] combine the existing first 4 and later 8 fixed-reference probes"
score_combined squared square_mean
score_combined absolute absolute_mean
score_combined probe_l2 term_root
score_combined timestamp_probe_l2 timestamp_root

JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_reference_timestamp_shared_probe4_scores.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --probe-seeds "${PROBE_SEEDS_CSV}" --combined-suffix "${COMBINED_SUFFIX}" \
  --combined-only --out-dir "${OUTDIR}"

echo "[done] cached fixed-reference 12-probe combinations: ${OUTDIR}"
