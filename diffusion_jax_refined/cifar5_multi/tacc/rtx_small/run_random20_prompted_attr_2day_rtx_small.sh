#!/usr/bin/env bash
#SBATCH -J cifar5-r20-rtx
#SBATCH -o cifar5-r20-rtx-%j.out
#SBATCH -e cifar5-r20-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_random_prompted_queries.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
  fi
fi

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-cifar5_multi_exp1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export LDS_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export DAS_PROJ_DIM="${DAS_PROJ_DIM:-4096}"
export DAS_DAMPING_SWEEP="${DAS_DAMPING_SWEEP:-1}"
export DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000}"
export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
export PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
export TRACIN_USE_SHARED_TRAIN_GRADIENT=1
export TRAJ_NUM_SNAPSHOTS="${TRAJ_NUM_SNAPSHOTS:-10}"
export TRAJ_TRAIN_MC_SAMPLES="${TRAJ_TRAIN_MC_SAMPLES:-10}"

echo "CIFAR5 multi random-20 prompted attribution RTX-small job"
echo "repo=${REPO_ROOT}"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; slots=${GPU_SLOTS:-2}"
echo "traj_namespace=${TRAJ_ATTRIBUTION_ARTIFACT_NAMESPACE:-h100_traj_ckptshared_10x10}"
echo "traj_score_query_normalize=${TRACIN_SCORE_QUERY_NORMALIZE:-0}; eps=${TRACIN_SCORE_QUERY_NORMALIZE_EPS:-1e-8}"
echo "das_damping_sweep_values=${DAS_DAMPING_SWEEP_VALUES}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

TRACIN_QN_ARGS=()
case "${TRACIN_SCORE_QUERY_NORMALIZE:-0}" in
  1|true|True|yes|Yes)
    TRACIN_QN_ARGS=(--tracin-score-query-normalize --tracin-score-query-normalize-eps "${TRACIN_SCORE_QUERY_NORMALIZE_EPS:-1e-8}")
    ;;
esac

"${PYTHON_BIN}" diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_random_prompted_queries.py \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --size "${CIFAR5_MULTI_SIZE:-10000}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --num-queries "${NUM_RANDOM_PROMPTED_QUERIES:-20}" \
  --random-query-seed "${RANDOM_QUERY_SEED:-0}" \
  --initial-seed-start "${INITIAL_SEED_START:-1000}" \
  --lds-m "${LDS_M:-64}" \
  --lds-percentage "${LDS_DATASET_PERCENTAGE:-25}" \
  --lds-subset-seeds "${LDS_SUBSET_SEEDS:-0,1,2}" \
  --gpus "${GPU_IDS:-0,1}" \
  --slots "${GPU_SLOTS:-2}" \
  --gpu-per-node 2 \
  --cpus-per-worker "${CPUS_PER_WORKER:-8}" \
  --slot-backend "${TACC_SLOT_BACKEND:-local}" \
  --max-parallel "${MAX_PARALLEL:-2}" \
  "${TRACIN_QN_ARGS[@]}" \
  ${EXTRA_CIFAR5_RANDOM20_ARGS:-}
