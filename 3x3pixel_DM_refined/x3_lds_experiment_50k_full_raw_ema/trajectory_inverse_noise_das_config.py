"""Configuration for query-dependent trajectory inverse-noise DAS."""

from exp_config import *


TRAJECTORY_INVERSE_DAS_QUERY_IDS = tuple(range(10))
TRAJECTORY_INVERSE_DAS_ALL_QUERY_IDS = tuple(range(100))
TRAJECTORY_INVERSE_DAS_FAMILY = "prompted"
TRAJECTORY_INVERSE_DAS_PROJ_DIM = 4096
TRAJECTORY_INVERSE_DAS_METHOD = (
    "das_ema_trajectory_inverse_noise_projected4096_99t_probe10_q00_q09"
)
TRAJECTORY_INVERSE_DAS_SHARD_DIR = (
    ATTR_DIR / "_trajectory_inverse_noise_das_99t_probe10_q00_q09_shards"
)
TRAJECTORY_INVERSE_DAS_100Q_METHOD = (
    "das_ema_trajectory_inverse_noise_projected4096_99t_probe10_100q"
)
TRAJECTORY_INVERSE_DAS_100Q_SHARD_DIR = (
    ATTR_DIR / "_trajectory_inverse_noise_das_99t_probe10_100q_shards"
)
TRAJECTORY_INVERSE_DAS_20T_POSITIONS = (
    1, 6, 12, 18, 24,
    25, 31, 37, 43, 49,
    50, 56, 62, 68, 74,
    75, 81, 87, 93, 99,
)
TRAJECTORY_INVERSE_DAS_20T_VALUES = tuple(
    DAS_TIMESTEPS[position]
    for position in TRAJECTORY_INVERSE_DAS_20T_POSITIONS
)
TRAJECTORY_INVERSE_DAS_20T_100Q_METHOD = (
    "das_ema_trajectory_inverse_noise_projected4096_20t_probe10_100q"
)
TRAJECTORY_INVERSE_DAS_20T_100Q_SHARD_DIR = (
    ATTR_DIR / "_trajectory_inverse_noise_das_20t_probe10_100q_shards"
)


def trajectory_inverse_das_shard_root(shard_index, shard_count):
    return TRAJECTORY_INVERSE_DAS_SHARD_DIR / (
        f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def trajectory_inverse_das_family_query_ids(family):
    if family == "prompted":
        return tuple(range(75))
    if family == "unprompted":
        return tuple(range(75, 100))
    raise ValueError(family)


def trajectory_inverse_das_100q_shard_root(
    family, query_shard_index, query_shard_count, timestamp_selection="99t"
):
    shard_dir = (
        TRAJECTORY_INVERSE_DAS_100Q_SHARD_DIR
        if timestamp_selection == "99t"
        else TRAJECTORY_INVERSE_DAS_20T_100Q_SHARD_DIR
    )
    return (
        shard_dir
        / family
        / f"query_shard_{int(query_shard_index):02d}_of_{int(query_shard_count):02d}"
    )


def trajectory_inverse_das_timestamp_indices(
    timestamp_selection, trajectory_timesteps=None
):
    if timestamp_selection == "99t":
        return tuple(range(99))
    if timestamp_selection == "20t":
        if trajectory_timesteps is None:
            raise ValueError("20t selection requires the cached trajectory timesteps")
        selected = tuple(
            min(
                range(len(trajectory_timesteps)),
                key=lambda index: abs(
                    int(trajectory_timesteps[index]) - int(target)
                ),
            )
            for target in TRAJECTORY_INVERSE_DAS_20T_VALUES
        )
        if len(set(selected)) != len(selected):
            raise ValueError("nearest cached trajectory timestamps are not unique")
        if any(int(trajectory_timesteps[index]) <= 0 for index in selected):
            raise ValueError("20t inverse-noise selection unexpectedly includes t=0")
        return selected
    raise ValueError(timestamp_selection)


def trajectory_inverse_das_100q_method(timestamp_selection):
    if timestamp_selection == "99t":
        return TRAJECTORY_INVERSE_DAS_100Q_METHOD
    if timestamp_selection == "20t":
        return TRAJECTORY_INVERSE_DAS_20T_100Q_METHOD
    raise ValueError(timestamp_selection)


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
