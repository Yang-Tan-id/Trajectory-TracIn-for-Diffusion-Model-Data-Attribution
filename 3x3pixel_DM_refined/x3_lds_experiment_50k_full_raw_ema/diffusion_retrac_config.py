"""Configuration for replayed Diffusion-TracIn and Diffusion-ReTrac."""

from exp_config import *


RETRAC_QUERY_IDS = tuple(range(100))
RETRAC_FAMILIES = ("prompted", "unprompted")
RETRAC_PARAM_SOURCE = "raw"
RETRAC_TIMESTEPS = tuple(
    int(round(1 + position * (T - 2) / 99.0)) for position in range(100)
)
RETRAC_QUERY_MC = 10
RETRAC_PROJ_DIM = 4096
RETRAC_EVENTS_PER_CHECKPOINT = BASE_SAVE_EVERY_EPOCHS
RETRAC_EPS = 1e-12

RETRAC_TRACIN_METHOD = (
    "diffusion_tracin_replayed_raw_50ckpt_100t_mc10_projected4096"
)
RETRAC_METHOD = (
    "diffusion_retrac_replayed_raw_50ckpt_100t_mc10_projected4096"
)
RETRAC_ADAMW_FULL_METHOD = (
    "diffusion_retrac_replayed_adamw_full_50ckpt_100t_mc10_projected4096"
)
RETRAC_METHODS = {
    "diffusion_tracin": RETRAC_TRACIN_METHOD,
    "diffusion_retrac": RETRAC_METHOD,
    "diffusion_retrac_adamw_full": RETRAC_ADAMW_FULL_METHOD,
}
RETRAC_SHARD_NAMESPACE = (
    "_diffusion_retrac_replayed_raw_adamw_full_100q_shards"
)


def retrac_query_ids(family):
    if family == "prompted":
        return tuple(range(75))
    if family == "unprompted":
        return tuple(range(75, 100))
    raise ValueError(family)


def retrac_shard_root(family, shard_index, shard_count):
    return (
        ATTR_DIR
        / RETRAC_SHARD_NAMESPACE
        / family
        / f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )
