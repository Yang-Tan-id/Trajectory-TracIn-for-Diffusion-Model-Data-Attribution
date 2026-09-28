"""Configuration for exact endpoint/noise-aligned TracIn-DAS."""

from exp_config import *


TRACIN_DAS_QUERY_IDS = tuple(range(10))
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


def tracin_das_shard_root(
    shard_index,
    shard_count,
    noise_mode="checkpoint",
    parameter_projection="exact",
    train_noise_mode="aligned",
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
    return (
        ATTR_DIR / namespace
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
