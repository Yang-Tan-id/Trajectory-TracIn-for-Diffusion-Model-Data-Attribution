"""Configuration for query-dependent trajectory-bridge noise alignment."""

from noise_pairing_ablation_config import *


TBA_QUERY_IDS = tuple(range(10))
TBA_DIRECTION_COUNT = 10
TBA_CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
TBA_VERSION = 1
TBA_PURE_IMPLIED_NOISE = False
TBA_TRAIN_NOISE_MODE = "aligned"


def tba_shard_root(shard_index, shard_count):
    return (
        ATTR_DIR
        / "_tracin_das_trajectory_bridge_aligned_10q_shards"
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def tba_method(contraction, group):
    return (
        "tracin_das_trajectory_bridge_aligned_"
        "10ckpt_20t_mc10_adamw_full_next_delta_projected4096_"
        f"raw_{contraction}_{group}"
    )
