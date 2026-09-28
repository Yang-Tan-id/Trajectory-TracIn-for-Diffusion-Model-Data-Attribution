"""Configuration for reference-trajectory forward-10 aligned-loss TracIn."""

from exp_config import *


REF_FORWARD10_DELTA_T = 10
REF_FORWARD10_NOISE_SEED = 51010
REF_FORWARD10_BATCH_SIZE = 512
REF_FORWARD10_METHODS = {
    "linear": (
        "traj_tracin_reference_forward10_next_delta_aligned_loss_projected4096_linear"
    ),
    "termwise_squared": (
        "traj_tracin_reference_forward10_next_delta_aligned_loss_projected4096_"
        "termwise_squared"
    ),
    "timestamp_sum_squared": (
        "traj_tracin_reference_forward10_next_delta_aligned_loss_projected4096_"
        "timestamp_sum_squared"
    ),
}


def ref_forward10_shard_root(family, shard_index, shard_count):
    if family not in FAMILIES:
        raise ValueError(f"unknown family={family!r}")
    return (
        ATTR_DIR
        / (
            "_traj_tracin_reference_forward10_next_delta_aligned_loss_"
            "projected4096_shards"
        )
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
