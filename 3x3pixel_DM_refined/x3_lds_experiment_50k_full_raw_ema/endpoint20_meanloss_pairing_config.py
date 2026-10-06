"""Endpoint-near trajectory-noise pairing with per-t or mean-noise loss."""

from noise_pairing_ablation_config import *


E20_QUERY_IDS = tuple(range(10))
E20_TIMESTAMP_POSITIONS = tuple(range(1, 21))
E20_TIMESTEPS = tuple(DAS_TIMESTEPS[position] for position in E20_TIMESTAMP_POSITIONS)
E20_MODES = ("per_timestamp_aligned", "mean_noise_mean_loss")
E20_CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
E20_VERSION = 1


def e20_shard_root(checkpoint_shard_index, checkpoint_shard_count):
    return (
        ATTR_DIR
        / "_tracin_das_endpoint20_inverse_noise_pairing_10q_shards"
        / f"checkpoint_shard_{int(checkpoint_shard_index):02d}_of_"
        f"{int(checkpoint_shard_count):02d}"
    )


def e20_method(mode, contraction):
    return (
        "tracin_das_endpoint20_inverse_noise_10ckpt_20t_"
        f"{mode}_adamw_full_next_delta_projected4096_raw_{contraction}_q00_q09"
    )
