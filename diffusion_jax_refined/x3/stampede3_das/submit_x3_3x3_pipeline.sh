#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_67}"
TRAIN_SEED="${TRAIN_SEED:-67}"
LDS_SEEDS="${LDS_SEEDS:-0 1 2}"
LDS_M="${LDS_M:-64}"
LDS_K="${LDS_K:-2500}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"

export EXPERIMENT_TAG TRAIN_SEED LDS_SEEDS LDS_M LDS_K PROJECTED_CACHE_DIM

cd "${REPO_ROOT}"

job1=$(sbatch --parsable \
  --export=ALL,EXPERIMENT_TAG="${EXPERIMENT_TAG}",TRAIN_SEED="${TRAIN_SEED}",LDS_SEEDS="${LDS_SEEDS}",LDS_M="${LDS_M}",LDS_K="${LDS_K}" \
  diffusion_jax_refined/x3/stampede3_das/00_train_base_and_lds_models_h100.sh)

job2=$(sbatch --parsable \
  --dependency=afterok:${job1} \
  --export=ALL,EXPERIMENT_TAG="${EXPERIMENT_TAG}",TRAIN_SEED="${TRAIN_SEED}",PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM}" \
  diffusion_jax_refined/x3/tacc/h100/projected_query48_rtx.sh)

job3=$(sbatch --parsable \
  --dependency=afterok:${job2} \
  --export=ALL,EXPERIMENT_TAG="${EXPERIMENT_TAG}",TRAIN_SEED="${TRAIN_SEED}",PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM}" \
  diffusion_jax_refined/x3/tacc/h100/projected_and_das_score_h100.sh)

printf 'base_lds_job=%s\nquery_job=%s\nscore_job=%s\n' "${job1}" "${job2}" "${job3}"
