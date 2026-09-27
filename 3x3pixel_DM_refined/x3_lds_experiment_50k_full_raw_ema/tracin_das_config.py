"""Configuration for exact endpoint/noise-aligned TracIn-DAS."""

from exp_config import *


TRACIN_DAS_QUERY_IDS = tuple(range(10))
TRACIN_DAS_FAMILY = "prompted"
TRACIN_DAS_BATCH_SIZE = 128
TRACIN_DAS_NOISE_SEED = 7367
TRACIN_DAS_DIRECTION_EPS = 1e-12
TRACIN_DAS_SHARD_NAMESPACE = "_tracin_das_endpoint_delta_checkpoint_noise_shards"
TRACIN_DAS_METHODS = {
    "linear": "tracin_das_endpoint_next_delta_checkpoint_noise_linear",
    "termwise_squared": "tracin_das_endpoint_next_delta_checkpoint_noise_termwise_squared",
    "timestamp_sum_squared": "tracin_das_endpoint_next_delta_checkpoint_noise_timestamp_sum_squared",
}


def tracin_das_shard_root(shard_index, shard_count):
    return (
        ATTR_DIR / TRACIN_DAS_SHARD_NAMESPACE
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
