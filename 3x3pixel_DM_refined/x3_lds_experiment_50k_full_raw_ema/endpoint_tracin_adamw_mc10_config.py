"""Endpoint-loss TracIn with 100 timestamps, MC10, and full AdamW."""

from exp_config import *


ETA_QUERY_IDS = tuple(range(100))
ETA_TIMESTAMPS = tuple(int(value) for value in DAS_TIMESTEPS)
ETA_QUERY_MC = 10
ETA_TRAIN_MC = 10
ETA_PROJECTION_DIM = 4096
ETA_EPS = 1e-12
ETA_CONTRACT_VERSION = 1
ETA_VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
ETA_CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")


def eta_method(variant, contraction):
    if variant not in ETA_VARIANTS:
        raise ValueError(variant)
    if contraction not in ETA_CONTRACTIONS:
        raise ValueError(contraction)
    return (
        "endpoint_tracin_simple_loss_100t_querymc10_trainmc10_"
        f"projected4096_adamw_full_{variant}_{contraction}"
    )


def eta_shard_root(family, shard_index, shard_count):
    return (
        ATTR_DIR
        / "_endpoint_tracin_adamw_full_100t_mc10_100q_shards"
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def eta_raw_gradient_method(variant, contraction, query_mc=1, train_mc=1):
    if variant not in ETA_VARIANTS:
        raise ValueError(variant)
    if contraction not in ETA_CONTRACTIONS:
        raise ValueError(contraction)
    return (
        f"endpoint_tracin_simple_loss_100t_querymc{int(query_mc)}_"
        f"trainmc{int(train_mc)}_projected4096_raw_gradient_"
        f"{variant}_{contraction}"
    )


def eta_raw_gradient_shard_root(
    family, shard_index, shard_count, query_mc=1, train_mc=1
):
    return (
        ATTR_DIR
        / (
            f"_endpoint_tracin_raw_gradient_100t_querymc{int(query_mc)}_"
            f"trainmc{int(train_mc)}_100q_shards"
        )
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
