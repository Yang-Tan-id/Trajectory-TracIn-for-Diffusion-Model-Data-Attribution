"""Controlled noise-pairing ablation for full-AdamW TracIn-DAS."""

from exp_config import *


NPA_QUERY_IDS = tuple(range(10))
NPA_FAMILY = "prompted"
NPA_DIRECTION_COUNT = 10
NPA_CHECKPOINT_PAIRS = tuple(
    int(round(position * 48.0 / 9.0)) for position in range(10)
)
NPA_TIMESTAMP_INDICES = tuple(
    quarter * 25 + offset
    for quarter in range(4)
    for offset in (0, 6, 12, 18, 24)
)
NPA_TIMESTAMP_GROUPS = {
    "all": NPA_TIMESTAMP_INDICES,
    **{
        f"q{quarter + 1}": tuple(
            index for index in NPA_TIMESTAMP_INDICES if index // 25 == quarter
        )
        for quarter in range(4)
    },
}
NPA_PAIRINGS = ("aligned", "cyclic", "random_permutation", "independent")
NPA_VARIANTS = ("raw", "query_train_l2")
NPA_CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
NPA_PROJECTION_DIM = 4096
NPA_NOISE_SEED = 20261004
NPA_EPS = 1e-12
NPA_CONTRACT_VERSION = 1


def npa_method(pairing, variant, contraction, timestamp_group):
    return (
        "tracin_das_noise_pairing_ablation_"
        "10ckpt_20t_mc10_adamw_full_next_delta_projected4096_"
        f"{pairing}_{variant}_{contraction}_{timestamp_group}"
    )


def npa_shard_root(timestamp_shard_index, timestamp_shard_count):
    return (
        ATTR_DIR
        / "_tracin_das_noise_pairing_ablation_10ckpt_20t_mc10_shards"
        / f"shard_{int(timestamp_shard_index):02d}_of_{int(timestamp_shard_count):02d}"
    )

