#!/usr/bin/env bash
set -euo pipefail

# Denominator-aware CelebA DAS. Train and query artifacts are independent of
# the leave-one-out correction, so reuse the completed full-10k EMA artifacts
# while writing scores and LDS reports to a distinct namespace.
export DAS_PARAMETER_SOURCE="ema"
export DAS_NAMESPACE="${DAS_NAMESPACE:-das_50x1_mc1_full10k_ema_sherman_morrison}"
export DAS_TRAIN_ARTIFACT_NAMESPACE="${DAS_TRAIN_ARTIFACT_NAMESPACE:-das_50x1_mc1_full10k_ema}"
export DAS_QUERY_ARTIFACT_NAMESPACE="${DAS_QUERY_ARTIFACT_NAMESPACE:-das_50x1_mc1_full10k_ema}"
export DAS_MAX_TRAIN_POINTS=10000
export DAS_SCORE_INDICES_PATH=""
export DAS_TOPK=10000
export DAS_SHERMAN_MORRISON_DENOMINATOR=1
export DAS_SCORE_DENOMINATOR_CACHE=1

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_das_50x1_lambda_sweep_4gpu.sh"
