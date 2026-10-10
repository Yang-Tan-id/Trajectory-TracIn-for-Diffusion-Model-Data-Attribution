"""Configuration for trajectory-predicted-noise aligned DAS."""

from exp_config import *


TPNA_DAS_QUERY_IDS = tuple(range(10))
TPNA_DAS_FAMILY = "prompted"
TPNA_DAS_PROJ_DIM = 4096
TPNA_DAS_METHOD = (
    "das_ema_trajectory_predicted_noise_aligned_projected4096_"
    "99t_probe10_q00_q09"
)
TPNA_DAS_SHARD_DIR = (
    ATTR_DIR
    / "_trajectory_predicted_noise_aligned_das_99t_probe10_q00_q09_shards"
)
TPNA_DAS_LDS_PATH = (
    LDS_DIR
    / "trajectory_predicted_noise_aligned_das_q00_q09_lambda_sweep.json"
)
TPNA_DAS_SEED = 978100


def tpna_das_shard_root(shard_index, shard_count):
    return TPNA_DAS_SHARD_DIR / (
        f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
