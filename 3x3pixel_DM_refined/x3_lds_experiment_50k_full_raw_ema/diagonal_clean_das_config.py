"""Configuration for timestamp-diagonal predicted-clean aligned DAS."""

from exp_config import *


DIAGONAL_CLEAN_QUERY_IDS = tuple(range(10))
DIAGONAL_CLEAN_FAMILY = "prompted"
DIAGONAL_CLEAN_CACHE_DIR = ROOT / "query_variants" / "predicted_clean_100timestamps"
DIAGONAL_CLEAN_METHOD = "das_ema_aligned_predicted_clean_100timestamp_diagonal"
DIAGONAL_CLEAN_SHARD_NAMESPACE = (
    "_das_ema_aligned_predicted_clean_100timestamp_diagonal_shards"
)


def diagonal_clean_shard_root(shard_index, shard_count):
    return (
        ATTR_DIR
        / DIAGONAL_CLEAN_SHARD_NAMESPACE
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
