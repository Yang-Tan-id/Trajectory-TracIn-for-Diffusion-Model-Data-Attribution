"""Configuration for query-dependent trajectory inverse-noise DAS."""

from exp_config import *


TRAJECTORY_INVERSE_DAS_QUERY_IDS = tuple(range(10))
TRAJECTORY_INVERSE_DAS_FAMILY = "prompted"
TRAJECTORY_INVERSE_DAS_PROJ_DIM = 4096
TRAJECTORY_INVERSE_DAS_METHOD = (
    "das_ema_trajectory_inverse_noise_projected4096_99t_probe10_q00_q09"
)
TRAJECTORY_INVERSE_DAS_SHARD_DIR = (
    ATTR_DIR / "_trajectory_inverse_noise_das_99t_probe10_q00_q09_shards"
)


def trajectory_inverse_das_shard_root(shard_index, shard_count):
    return TRAJECTORY_INVERSE_DAS_SHARD_DIR / (
        f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
