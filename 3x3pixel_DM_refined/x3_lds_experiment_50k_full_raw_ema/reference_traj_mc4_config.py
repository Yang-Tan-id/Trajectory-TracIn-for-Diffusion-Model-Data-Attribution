"""Reference-trajectory state-perturbation MC4 TracIn-DAS configuration."""

from exp_config import *


REF_MC4_QUERY_IDS = tuple(range(10))
REF_MC4_FAMILY = "prompted"
REF_MC4_COUNT = 4
REF_MC4_DEFAULT_EPSILON = 0.01
REF_MC4_DIRECTION_SEED = 40767
REF_MC4_BATCH_SIZE = 128
REF_MC4_DIRECTION_EPS = 1e-12
REF_MC4_DEFAULT_TRAIN_MC = 1


def epsilon_tag(epsilon):
    return format(float(epsilon), ".8g").replace("-", "m").replace(".", "p")


def ref_mc4_methods(epsilon, train_mc=REF_MC4_DEFAULT_TRAIN_MC):
    prefix = (
        f"tracin_das_reference_traj_mc4_eps_{epsilon_tag(epsilon)}_"
        f"projected4096_train_mc{int(train_mc)}"
    )
    return {
        "linear": f"{prefix}_linear",
        "termwise_squared": f"{prefix}_termwise_squared",
        "timestamp_sum_squared": f"{prefix}_timestamp_sum_squared",
    }


def ref_mc4_shard_root(
    shard_index,
    shard_count,
    epsilon,
    train_mc=REF_MC4_DEFAULT_TRAIN_MC,
):
    return (
        ATTR_DIR
        / (
            f"_tracin_das_reference_traj_mc4_eps_{epsilon_tag(epsilon)}_"
            f"projected4096_train_mc{int(train_mc)}_shards"
        )
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
