"""Configuration for direction-aligned, all-t integrated TracIn-DAS."""

from exp_config import *


DITD_QUERY_IDS = tuple(range(10))
DITD_FAMILY = "prompted"
DITD_DIRECTION_COUNT = 100
DITD_TRAIN_TIMESTEPS = tuple(range(T))
DITD_QUERY_TIMESTEPS = tuple(int(value) for value in DAS_TIMESTEPS)
DITD_PROJECTION_DIM = 4096
DITD_NOISE_SEED = 20261002
DITD_DIRECTION_EPS = 1e-12
DITD_CONTRACT_VERSION = 1

DITD_METHOD_STEM = (
    "tracin_das_endpoint_next_delta_direction_aligned_"
    "train_all1000t_projected4096_100dir"
)
DITD_METHODS = {
    "linear": f"{DITD_METHOD_STEM}_linear",
    "termwise_squared": f"{DITD_METHOD_STEM}_termwise_squared",
    "timestamp_sum_squared": f"{DITD_METHOD_STEM}_timestamp_sum_squared",
}


def ditd_shard_root(shard_index, shard_count):
    return (
        ATTR_DIR
        / f"_{DITD_METHOD_STEM}_shards"
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )

