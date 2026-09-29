"""Configuration for ten-anchor predicted-clean-image aligned DAS."""

from exp_config import *


MULTICLEAN_QUERY_IDS = tuple(range(100))
MULTICLEAN_FAMILIES = ("prompted", "unprompted")
MULTICLEAN_ANCHOR_INDICES = (99, 88, 77, 66, 55, 44, 33, 22, 11, 0)
MULTICLEAN_ANCHOR_COUNT = len(MULTICLEAN_ANCHOR_INDICES)
MULTICLEAN_ANCHOR_DAS_COUNTS = tuple(
    10 * (index + 1) for index in range(MULTICLEAN_ANCHOR_COUNT)
)
MULTICLEAN_CACHE_DIR = (
    ROOT / "query_variants" / "predicted_clean_10anchors_triangular_100q"
)
MULTICLEAN_METHOD = "das_ema_aligned_predicted_clean_10anchors_triangular_sum"
MULTICLEAN_SHARD_NAMESPACE = (
    "_das_ema_aligned_predicted_clean_10anchors_triangular_sum_100q_shards"
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
