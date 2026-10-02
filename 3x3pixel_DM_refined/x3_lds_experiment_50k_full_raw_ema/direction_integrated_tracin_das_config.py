"""Configuration for direction-aligned, all-t integrated TracIn-DAS."""

from exp_config import *


DITD_QUERY_IDS = tuple(range(10))
DITD_FAMILY = "prompted"
DITD_DIRECTION_COUNT = 100
DITD_DEFAULT_TRAIN_T_COUNT = 100
DITD_QUERY_TIMESTEPS = tuple(int(value) for value in DAS_TIMESTEPS)
DITD_PROJECTION_DIM = 4096
DITD_NOISE_SEED = 20261002
DITD_DIRECTION_EPS = 1e-12
DITD_CONTRACT_VERSION = 1

def ditd_train_timesteps(count):
    count = int(count)
    if count == T:
        return tuple(range(T))
    if count < 2 or count > T:
        raise ValueError(f"train timestep count must be in [2,{T}], got {count}")
    values = tuple(
        int(round(position * (T - 1) / (count - 1)))
        for position in range(count)
    )
    if len(set(values)) != count:
        raise ValueError("rounded train timestamps are not unique")
    return values


def ditd_method_stem(train_t_count):
    return (
        "tracin_das_endpoint_next_delta_direction_aligned_"
        f"train_{int(train_t_count)}t_even_projected4096_100dir"
    )


def ditd_methods(train_t_count):
    stem = ditd_method_stem(train_t_count)
    return {
        "linear": f"{stem}_linear",
        "termwise_squared": f"{stem}_termwise_squared",
        "timestamp_sum_squared": f"{stem}_timestamp_sum_squared",
    }


def ditd_shard_root(shard_index, shard_count, train_t_count):
    return (
        ATTR_DIR
        / f"_{ditd_method_stem(train_t_count)}_shards"
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
