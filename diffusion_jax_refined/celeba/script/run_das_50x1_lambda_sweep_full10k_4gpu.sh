#!/usr/bin/env bash
set -euo pipefail

# Full-training-set CelebA DAS.  The empty index path deliberately disables the
# historical attribution_5k_indices.npy restriction.  Query gradients do not
# depend on the candidate training set, so reuse the already computed 5k-run
# query artifacts while writing all train features and scores to a new namespace.
export DAS_NAMESPACE="${DAS_NAMESPACE:-das_50x1_mc1_full10k}"
export DAS_QUERY_ARTIFACT_NAMESPACE="${DAS_QUERY_ARTIFACT_NAMESPACE:-das_50x1_mc1}"
export DAS_MAX_TRAIN_POINTS=10000
export DAS_SCORE_INDICES_PATH=""
export DAS_TOPK=10000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_das_50x1_lambda_sweep_4gpu.sh"
