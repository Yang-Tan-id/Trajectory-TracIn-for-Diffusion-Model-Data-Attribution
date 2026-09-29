"""Configuration for ten-anchor predicted-clean-image aligned DAS."""

from exp_config import *


MULTICLEAN_QUERY_IDS = tuple(range(100))
MULTICLEAN_FAMILIES = ("prompted", "unprompted")
MULTICLEAN_ANCHOR_INDICES = (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)
MULTICLEAN_ANCHOR_COUNT = len(MULTICLEAN_ANCHOR_INDICES)
MULTICLEAN_CACHE_DIR = ROOT / "query_variants" / "predicted_clean_10anchors_100q"
MULTICLEAN_METHOD = "das_ema_aligned_predicted_clean_10anchors_sum"
MULTICLEAN_SHARD_NAMESPACE = (
    "_das_ema_aligned_predicted_clean_10anchors_sum_100q_shards"
)


def multiclean_query_ids(family):
    if family == "prompted":
        return tuple(range(75))
    if family == "unprompted":
        return tuple(range(75, 100))
    raise ValueError(family)


def multiclean_shard_root(family, shard_index, shard_count):
    return (
        ATTR_DIR
        / MULTICLEAN_SHARD_NAMESPACE
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
