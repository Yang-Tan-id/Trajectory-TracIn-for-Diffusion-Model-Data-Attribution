"""Configuration for trajectory-state inverse-noise TracIn-DAS."""

from exp_config import *


INVERSE_NOISE_QUERY_IDS = tuple(range(10))
INVERSE_NOISE_FAMILY = "prompted"
INVERSE_NOISE_PROJ_DIM = 4096
INVERSE_NOISE_CONTRACTIONS = (
    "linear",
    "termwise_squared",
    "timestamp_sum_squared",
)
INVERSE_NOISE_SUFFIX = "no_endpoint_q00_q09"
INVERSE_NOISE_METHODS = {
    contraction: (
        "tracin_das_trajectory_inverse_noise_projected4096_"
        f"{contraction}_{INVERSE_NOISE_SUFFIX}"
    )
    for contraction in INVERSE_NOISE_CONTRACTIONS
}
INVERSE_NOISE_SHARD_DIR = (
    ATTR_DIR / "_tracin_das_trajectory_inverse_noise_projected4096_shards"
)


def inverse_noise_shard_root(shard_index, shard_count):
    return INVERSE_NOISE_SHARD_DIR / (
        f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
