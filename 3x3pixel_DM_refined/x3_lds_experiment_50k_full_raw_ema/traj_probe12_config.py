"""Configuration for the q00-q19 twelve-output-probe Traj experiment."""

from exp_config import *


TRAJ_PROBE12_QUERY_IDS = tuple(range(20))
TRAJ_PROBE12_FAMILY = "prompted"
TRAJ_PROBE12_NUM_PROBES = 12
TRAJ_PROBE12_PROBE_SEED = 12067
TRAJ_PROBE12_CHECKPOINT_COUNT = 49
TRAJ_PROBE12_BATCH_SIZE = 128
TRAJ_PROBE12_QUERY_NORM_EPS = 1e-8
TRAJ_PROBE12_SHARD_NAMESPACE = "_traj_probe12_timestamp_shards"

TRAJ_PROBE12_METHODS = {
    ("termwise_squared", "none"): (
        "traj_probe12_first_raw_termwise_squared"
    ),
    ("termwise_squared", "query_l2"): (
        "traj_probe12_first_raw_termwise_squared_query_l2"
    ),
    ("timestamp_sum_squared", "none"): (
        "traj_probe12_first_raw_timestamp_sum_squared"
    ),
    ("timestamp_sum_squared", "query_l2"): (
        "traj_probe12_first_raw_timestamp_sum_squared_query_l2"
    ),
}

TRAJ_PROBE12_VARIANTS = (
    ("termwise_squared", "none"),
    ("termwise_squared", "query_l2"),
    ("timestamp_sum_squared", "none"),
    ("timestamp_sum_squared", "query_l2"),
)


def traj_probe12_shard_root(shard_index, shard_count):
    return (
        ATTR_DIR
        / TRAJ_PROBE12_SHARD_NAMESPACE
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
