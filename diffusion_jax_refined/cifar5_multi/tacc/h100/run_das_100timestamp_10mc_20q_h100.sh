#!/usr/bin/env bash
#SBATCH -J cifar5-das-100ts-20q-h100
#SBATCH -o cifar5-das-100ts-20q-h100-%j.out
#SBATCH -e cifar5-das-100ts-20q-h100-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
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
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh. Set ENV_SETUP=/path/to/env_setup.sh if needed." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_NAME:-experiment_dm}"
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
export DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000}"
export DAS_TIMESTEPS="${DAS_TIMESTEPS:-0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990}"
export DAS_NUM_MC_NOISE="${DAS_NUM_MC_NOISE:-10}"
export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
export PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
export TRACIN_USE_SHARED_TRAIN_GRADIENT="${TRACIN_USE_SHARED_TRAIN_GRADIENT:-1}"
export ATTRIBUTION_ARTIFACT_NAMESPACE="${ATTRIBUTION_ARTIFACT_NAMESPACE:-h100_das_100ts_20q_0p1-50000}"
export TRAJ_NUM_SNAPSHOTS="${TRAJ_NUM_SNAPSHOTS:-10}"
export TRAJ_TRAIN_MC_SAMPLES="${TRAJ_TRAIN_MC_SAMPLES:-10}"

echo "CIFAR5 multi DAS 100-timestamp x10 random-20 H100 job"
echo "repo=${REPO_ROOT}"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; slots=${GPU_SLOTS:-16}"
echo "das_namespace=${ATTRIBUTION_ARTIFACT_NAMESPACE}"
echo "das_timesteps_count=100"
echo "das_num_mc_noise=${DAS_NUM_MC_NOISE}; das_damping_sweep_values=${DAS_DAMPING_SWEEP_VALUES}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN}" diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_random_prompted_queries.py \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --size "${CIFAR5_MULTI_SIZE:-10000}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --num-queries "${NUM_RANDOM_PROMPTED_QUERIES:-20}" \
  --random-query-seed "${RANDOM_QUERY_SEED:-0}" \
  --initial-seed-start "${INITIAL_SEED_START:-1000}" \
  --extra-prompted-queries "${EXTRA_PROMPTED_QUERIES:-horse,automobile,cat;horse,dog,cat}" \
  --extra-initial-seed "${EXTRA_INITIAL_SEED:-0}" \
  --traj-artifact-namespace "${TRAJ_ATTRIBUTION_ARTIFACT_NAMESPACE:-h100_traj_ckptshared_10x10}" \
  --skip-traj-tracin \
  --skip-lds-eval \
  --gpus "${GPU_IDS:-0,1,2,3}" \
  --slots "${GPU_SLOTS:-16}" \
  --gpu-per-node 4 \
  --cpus-per-worker "${CPUS_PER_WORKER:-24}" \
  --slot-backend "${TACC_SLOT_BACKEND:-ibrun}" \
  --max-parallel "${MAX_PARALLEL:-16}" \
  ${EXTRA_CIFAR5_RANDOM20_ARGS:-}
