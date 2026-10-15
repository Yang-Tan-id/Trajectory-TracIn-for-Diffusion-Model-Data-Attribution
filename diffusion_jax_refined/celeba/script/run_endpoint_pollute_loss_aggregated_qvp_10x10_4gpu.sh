#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOSS_SIDE_AGGREGATE_QVP=1
export QUERY_DIRECTION_COUNT=1
export TIMESTEPS="${TIMESTEPS:-999,888,777,666,555,444,333,222,111,0}"
export TIMESTEP_TAG="${TIMESTEP_TAG:-uniform10}"
export CHECKPOINT_INDICES="${CHECKPOINT_INDICES:-0,5,10,16,21,26,32,37,42,48}"

exec bash "$script_dir/run_endpoint_pollute_multidelta_10t_4gpu.sh"
