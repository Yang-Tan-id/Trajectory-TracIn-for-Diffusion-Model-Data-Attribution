#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOSS_SIDE_AGGREGATE_QVP=1
export AGGREGATE_ONLY=1
export QUERY_DIRECTION_COUNT=1
# Exact model-evaluation schedule used by the 50-step DDIM sampler:
# inference_timesteps(1000, 50) == range(999, -1, -20).
export TIMESTEPS="${TIMESTEPS:-999,979,959,939,919,899,879,859,839,819,799,779,759,739,719,699,679,659,639,619,599,579,559,539,519,499,479,459,439,419,399,379,359,339,319,299,279,259,239,219,199,179,159,139,119,99,79,59,39,19}"
export TIMESTEP_TAG="${TIMESTEP_TAG:-ddim50}"
# Fifty saved checkpoints define 49 adjacent source -> target transitions.
export CHECKPOINT_INDICES="${CHECKPOINT_INDICES:-$(seq -s, 0 48)}"

exec bash "$script_dir/run_endpoint_pollute_multidelta_10t_4gpu.sh"
