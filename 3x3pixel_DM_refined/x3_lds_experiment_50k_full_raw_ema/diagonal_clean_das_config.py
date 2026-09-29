"""Configuration for timestamp-diagonal predicted-clean aligned DAS."""

from exp_config import *


DIAGONAL_CLEAN_QUERY_IDS = tuple(range(100))
DIAGONAL_CLEAN_FAMILIES = ("prompted", "unprompted")
DIAGONAL_CLEAN_CACHE_DIR = (
    ROOT / "query_variants" / "predicted_clean_100timestamps_100q"
)
DIAGONAL_CLEAN_METHOD = "das_ema_aligned_predicted_clean_100timestamp_diagonal"
DIAGONAL_CLEAN_10_INDICES = (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)
DIAGONAL_CLEAN_10_METHOD = "das_ema_aligned_predicted_clean_10timestamp_diagonal"
DIAGONAL_CLEAN_SHARD_NAMESPACE = (
    "_das_ema_aligned_predicted_clean_100timestamp_diagonal_100q_shards"
)


def diagonal_clean_query_ids(family):
    if family == "prompted":
        return tuple(range(75))
    if family == "unprompted":
        return tuple(range(75, 100))
    raise ValueError(family)


def diagonal_clean_method(timestamp_count):
    if int(timestamp_count) == 100:
        return DIAGONAL_CLEAN_METHOD
    if int(timestamp_count) == 10:
        return DIAGONAL_CLEAN_10_METHOD
    raise ValueError(timestamp_count)


def diagonal_clean_indices(timestamp_count):
    if int(timestamp_count) == 100:
        return tuple(range(100))
    if int(timestamp_count) == 10:
        return DIAGONAL_CLEAN_10_INDICES
    raise ValueError(timestamp_count)


def diagonal_clean_shard_root(family, shard_index, shard_count, timestamp_count=100):
    return (
        ATTR_DIR
        / DIAGONAL_CLEAN_SHARD_NAMESPACE
        / f"{int(timestamp_count)}timestamps"
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
