#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOSS_SIDE_AGGREGATE_QVP=1
export AGGREGATE_ONLY=1
export QUERY_DIRECTION_COUNT=1
export TIMESTEPS="${TIMESTEPS:-999,978,958,937,917,897,876,856,835,815,795,774,754,733,713,693,672,652,632,611,591,570,550,530,509,489,468,448,428,407,387,366,346,326,305,285,265,244,224,203,183,163,142,122,101,81,61,40,20,0}"
export TIMESTEP_TAG="${TIMESTEP_TAG:-uniform50}"
# Fifty saved checkpoints define 49 adjacent source -> target transitions.
export CHECKPOINT_INDICES="${CHECKPOINT_INDICES:-$(seq -s, 0 48)}"

exec bash "$script_dir/run_endpoint_pollute_multidelta_10t_4gpu.sh"
