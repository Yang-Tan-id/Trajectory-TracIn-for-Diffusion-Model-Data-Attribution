"""Configuration for vector-valued, interaction-aware Bundle TracIn."""

from exp_config import *


BUNDLE_TRACIN_QUERY_IDS = tuple(range(10))
BUNDLE_TRACIN_TRAIN_MC = 10
BUNDLE_TRACIN_PROJ_DIM = 4096
BUNDLE_TRACIN_PARAM_SOURCE = "raw"
BUNDLE_TRACIN_BATCH_SIZE = TRACIN_PROJECTED_BATCH_SIZE
BUNDLE_TRACIN_CHECKPOINT_SHARDS = 4
BUNDLE_TRACIN_OUTPUT_DIM = 3 * 3 * 3
BUNDLE_TRACIN_METHOD = (
    "bundle_tracin_raw_mc10_independent_t_noise_projected4096_"
    "checkpoint_sum_timestamp_mean"
)
BUNDLE_TRACIN_SHARD_NAMESPACE = f"_{BUNDLE_TRACIN_METHOD}_checkpoint_shards"


def parse_query_ids(value):
    """Parse comma-separated IDs and inclusive ranges such as ``0-9,20``."""
    if value is None:
        return list(BUNDLE_TRACIN_QUERY_IDS)
    result = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"invalid query range: {token}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    if not result:
        raise ValueError("query list must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("query list contains duplicates")
    if any(query_id < 0 or query_id >= len(INITIAL_SEEDS) for query_id in result):
        raise ValueError(f"query IDs must be in [0, {len(INITIAL_SEEDS) - 1}]")
    return result
