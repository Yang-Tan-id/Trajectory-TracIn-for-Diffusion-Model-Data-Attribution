"""Configuration for trajectory-state relative-forward aligned DAS."""

from exp_config import *


MULTICLEAN_QUERY_IDS = tuple(range(100))
MULTICLEAN_FAMILIES = ("prompted", "unprompted")
MULTICLEAN_ANCHOR_INDICES = (11, 22, 33, 44, 55, 66, 77, 88, 99)
MULTICLEAN_ANCHOR_COUNT = len(MULTICLEAN_ANCHOR_INDICES)
MULTICLEAN_ANCHOR_DAS_COUNTS = tuple(
    10 * (index + 1) for index in range(MULTICLEAN_ANCHOR_COUNT)
)
MULTICLEAN_CACHE_DIR = (
    ROOT / "query_variants" / "trajectory_state_relative_forward_9anchors_100q"
)
MULTICLEAN_METHOD = "das_ema_aligned_trajectory_state_relative_forward_9anchors_sum"
MULTICLEAN_SHARD_NAMESPACE = (
    "_das_ema_aligned_trajectory_state_relative_forward_9anchors_sum_100q_shards"
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


def multiclean_anchor_targets(anchor_timestep, target_count):
    if target_count <= 0:
        raise ValueError(target_count)
    if not 0 <= anchor_timestep < T:
        raise ValueError(anchor_timestep)
    if target_count == 1:
        return (int(anchor_timestep),)
    return tuple(
        int(anchor_timestep + index * (T - 1 - anchor_timestep) / (target_count - 1))
        for index in range(target_count)
    )


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
