"""X3 direction20/mean100t full-AdamW trajectory attribution config."""

from exp_config import *


D20_DIRECTION_COUNT = 20
D20_TRAIN_TIMESTAMPS = tuple(int(value) for value in DAS_TIMESTEPS)
D20_QUERY_TIMESTAMPS = tuple(int(value) for value in DAS_TIMESTEPS)
D20_PROJECTION_DIM = 4096
D20_NOISE_SEED = 20261003
D20_EPS = 1e-12
D20_CONTRACT_VERSION = 1
D20_QUERY_MODES = ("trajectory", "endpoint")
D20_VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def d20_method(
    mode,
    variant,
    noise_pairing="aligned-shared-direction",
    direction_count=D20_DIRECTION_COUNT,
):
    if mode not in D20_QUERY_MODES:
        raise ValueError(mode)
    if variant not in D20_VARIANTS:
        raise ValueError(variant)
    if noise_pairing == "aligned-shared-direction":
        query = (
            "reference_trajectory_nonaligned_noise"
            if mode == "trajectory"
            else "endpoint_polluted_direction_aligned"
        )
        experiment = f"direction{int(direction_count)}_mean100t"
    elif noise_pairing == "independent-per-t":
        query = (
            "reference_trajectory_independent_mean100t_loss"
            if mode == "trajectory"
            else "endpoint_polluted_independent_mean100t_loss"
        )
        experiment = f"bank{int(direction_count)}_100x1"
    else:
        raise ValueError(noise_pairing)
    prefix = "traj_tracin" if mode == "trajectory" else "tracin_das"
    return (
        f"{prefix}_{experiment}_adamw_full_{query}_"
        f"next_delta_normalized_projected4096_{variant}_"
        "timestamp_sum_squared"
    )


def d20_shard_root(
    family,
    shard_index,
    shard_count,
    noise_pairing="aligned-shared-direction",
    direction_count=D20_DIRECTION_COUNT,
    query_modes=D20_QUERY_MODES,
):
    if noise_pairing == "aligned-shared-direction" and int(direction_count) == 20:
        experiment = "direction20_mean100t_adamw_full_dual"
    elif noise_pairing == "independent-per-t":
        modes = "_".join(query_modes)
        experiment = f"bank{int(direction_count)}_100x1_adamw_full_{modes}_independent"
    else:
        experiment = f"direction{int(direction_count)}_mean100t_adamw_full_dual"
    return (
        ATTR_DIR
        / f"_{experiment}_100q_shards"
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
