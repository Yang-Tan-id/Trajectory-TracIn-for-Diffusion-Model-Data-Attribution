"""Configuration for exact endpoint/noise-aligned TracIn-DAS."""

from exp_config import *


TRACIN_DAS_QUERY_IDS = tuple(range(10))
TRACIN_DAS_ALL_QUERY_IDS = tuple(range(100))
TRACIN_DAS_FIRST99_QUERY_IDS = tuple(range(99))
TRACIN_DAS_FAMILY = "prompted"
TRACIN_DAS_BATCH_SIZE = 128
TRACIN_DAS_NOISE_SEED = 7367
TRACIN_DAS_DIRECTION_EPS = 1e-12
TRACIN_DAS_NOISE_MODES = ("checkpoint", "timestamp-shared")
TRACIN_DAS_PARAMETER_PROJECTIONS = ("exact", "projected4096")
TRACIN_DAS_TRAIN_NOISE_MODES = ("aligned", "independent-mc10")
TRACIN_DAS_METHODS_BY_VARIANT = {
    ("checkpoint", "exact"): {
        "linear": "tracin_das_endpoint_next_delta_checkpoint_noise_linear",
        "termwise_squared": "tracin_das_endpoint_next_delta_checkpoint_noise_termwise_squared",
        "timestamp_sum_squared": "tracin_das_endpoint_next_delta_checkpoint_noise_timestamp_sum_squared",
    },
    ("timestamp-shared", "exact"): {
        "linear": "tracin_das_endpoint_next_delta_timestamp_shared_noise_linear",
        "termwise_squared": "tracin_das_endpoint_next_delta_timestamp_shared_noise_termwise_squared",
        "timestamp_sum_squared": "tracin_das_endpoint_next_delta_timestamp_shared_noise_timestamp_sum_squared",
    },
    ("checkpoint", "projected4096"): {
        "linear": "tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_linear",
        "termwise_squared": "tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_termwise_squared",
        "timestamp_sum_squared": "tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_timestamp_sum_squared",
    },
    ("timestamp-shared", "projected4096"): {
        "linear": "tracin_das_endpoint_next_delta_timestamp_shared_noise_projected4096_linear",
        "termwise_squared": "tracin_das_endpoint_next_delta_timestamp_shared_noise_projected4096_termwise_squared",
        "timestamp_sum_squared": "tracin_das_endpoint_next_delta_timestamp_shared_noise_projected4096_timestamp_sum_squared",
    },
}
TRACIN_DAS_METHODS = TRACIN_DAS_METHODS_BY_VARIANT[("checkpoint", "exact")]
TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS = (50, 40, 25, 20, 15, 10, 5)
TRACIN_DAS_TIMESTAMP_COUNTS = tuple(range(100, 0, -10))


def tracin_das_methods(
    noise_mode,
    parameter_projection="exact",
    train_noise_mode="aligned",
):
    methods = TRACIN_DAS_METHODS_BY_VARIANT[
        (str(noise_mode), str(parameter_projection))
    ]
    if train_noise_mode == "aligned":
        return methods
    return {
        contraction: (
            method[: -len(contraction)] + "train_mc10_" + contraction
        )
        for contraction, method in methods.items()
    }


def tracin_das_avg_pair_lr_methods(checkpoint_count):
    checkpoint_count = int(checkpoint_count)
    if checkpoint_count not in TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS:
        raise ValueError(f"unsupported checkpoint_count={checkpoint_count}")
    stem = (
        "tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_"
        f"interval_mean_lr_{checkpoint_count}ckpt"
    )
    return {
        "linear": f"{stem}_linear",
        "termwise_squared": f"{stem}_termwise_squared",
        "timestamp_sum_squared": f"{stem}_timestamp_sum_squared",
    }


def tracin_das_checkpoint_pair_indices(checkpoint_count):
    """Even source-transition positions, always including position 48."""
    checkpoint_count = int(checkpoint_count)
    if checkpoint_count == 50:
        return tuple(range(49))
    if checkpoint_count not in TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS:
        raise ValueError(f"unsupported checkpoint_count={checkpoint_count}")
    # This matches the existing 10-interval contract exactly:
    # (0, 5, 11, 16, 21, 27, 32, 37, 43, 48).
    return tuple(
        int(round(position * 48.0 / (checkpoint_count - 1)))
        for position in range(checkpoint_count)
    )


def tracin_das_avg_pair_lr_shard_root(family, shard_index, shard_count):
    if family not in FAMILIES:
        raise ValueError(f"unknown family={family!r}")
    return (
        ATTR_DIR
        / "_tracin_das_checkpoint_noise_projected4096_interval_mean_lr_99q_multi_ckpt_shards"
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def tracin_das_interval_mean_lr_timestamp_methods(timestamp_count):
    timestamp_count = int(timestamp_count)
    if timestamp_count not in TRACIN_DAS_TIMESTAMP_COUNTS:
        raise ValueError(f"unsupported timestamp_count={timestamp_count}")
    stem = (
        "tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_"
        f"interval_mean_lr_50ckpt_{timestamp_count}timestamp"
    )
    return {
        "linear": f"{stem}_linear",
        "termwise_squared": f"{stem}_termwise_squared",
        "timestamp_sum_squared": f"{stem}_timestamp_sum_squared",
    }


def tracin_das_timestamp_indices(timestamp_count):
    """Even positions among 100 timestamps, including positions 0 and 99."""
    timestamp_count = int(timestamp_count)
    if timestamp_count == 100:
        return tuple(range(100))
    if timestamp_count not in TRACIN_DAS_TIMESTAMP_COUNTS:
        raise ValueError(f"unsupported timestamp_count={timestamp_count}")
    return tuple(
        int(round(position * 99.0 / (timestamp_count - 1)))
        for position in range(timestamp_count)
    )


def tracin_das_timestamp_sweep_shard_root(family, shard_index, shard_count):
    if family not in FAMILIES:
        raise ValueError(f"unknown family={family!r}")
    return (
        ATTR_DIR
        / "_tracin_das_checkpoint_noise_projected4096_interval_mean_lr_99q_timestamp_sweep_shards"
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def tracin_das_shard_root(
    shard_index,
    shard_count,
    noise_mode="checkpoint",
    parameter_projection="exact",
    train_noise_mode="aligned",
    family=None,
    query_scope="ten",
):
    noise_tag = (
        "checkpoint_noise" if noise_mode == "checkpoint"
        else "timestamp_shared_noise"
    )
    projection_tag = "" if parameter_projection == "exact" else "_projected4096"
    train_tag = "" if train_noise_mode == "aligned" else "_train_mc10"
    namespace = (
        f"_tracin_das_endpoint_delta_{noise_tag}{projection_tag}"
        f"{train_tag}_shards"
    )
    root = ATTR_DIR / namespace
    if query_scope == "all":
        if family not in FAMILIES:
            raise ValueError("family is required for the all-query shard bank")
        root = root / "q00_q99" / str(family)
    elif query_scope != "ten":
        raise ValueError(f"unknown query_scope={query_scope!r}")
    return root / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
