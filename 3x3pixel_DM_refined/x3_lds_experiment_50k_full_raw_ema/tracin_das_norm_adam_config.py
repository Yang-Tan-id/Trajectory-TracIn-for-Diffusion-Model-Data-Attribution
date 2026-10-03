"""Four gradient-normalization variants for raw and full-AdamW TracIn-DAS."""

from exp_config import *


TDNA_VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TDNA_TRANSFORMS = ("gradient", "adamw_full")
TDNA_CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
TDNA_EPS = 1e-12
TDNA_CONTRACT_VERSION = 1
TDNA_MC10 = 10
TDNA_MC10_CONTRACT_VERSION = 1


def tdna_method(transform, variant, contraction):
    if transform not in TDNA_TRANSFORMS:
        raise ValueError(transform)
    if variant not in TDNA_VARIANTS:
        raise ValueError(variant)
    if contraction not in TDNA_CONTRACTIONS:
        raise ValueError(contraction)
    transform_tag = "gradient" if transform == "gradient" else "adamw_full"
    return (
        "tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_"
        f"{transform_tag}_{variant}_{contraction}"
    )


def tdna_methods():
    return {
        transform: {
            variant: {
                contraction: tdna_method(transform, variant, contraction)
                for contraction in TDNA_CONTRACTIONS
            }
            for variant in TDNA_VARIANTS
        }
        for transform in TDNA_TRANSFORMS
    }


def tdna_shard_root(family, timestamp_shard_index, timestamp_shard_count):
    if family not in FAMILIES:
        raise ValueError(family)
    return (
        ATTR_DIR
        / "_tracin_das_norm4_adamw_full_100q_shards"
        / family
        / f"shard_{int(timestamp_shard_index):02d}_of_{int(timestamp_shard_count):02d}"
    )


def tdna_mc10_method(variant):
    if variant not in TDNA_VARIANTS:
        raise ValueError(variant)
    return (
        "tracin_das_endpoint_next_delta_normalized_checkpoint_noise_"
        "projected4096_adamw_full_aligned_mc10_"
        f"{variant}_timestamp_sum_squared"
    )


def tdna_mc10_methods():
    return {variant: tdna_mc10_method(variant) for variant in TDNA_VARIANTS}


def tdna_mc10_shard_root(family, timestamp_shard_index, timestamp_shard_count):
    if family not in FAMILIES:
        raise ValueError(family)
    return (
        ATTR_DIR
        / "_tracin_das_adamw_full_delta_norm_aligned_mc10_100q_shards"
        / family
        / f"shard_{int(timestamp_shard_index):02d}_of_{int(timestamp_shard_count):02d}"
    )
