"""Configuration for fully-unrolled trajectory-response DAS."""

from exp_config import *


UNROLLED_TRAJ_DAS_QUERY_IDS = tuple(range(10))
UNROLLED_TRAJ_DAS_FAMILY = "prompted"
UNROLLED_TRAJ_DAS_OUTPUT_DIM = 9
UNROLLED_TRAJ_DAS_PROJECTION_DIM = 4096
UNROLLED_TRAJ_DAS_PROJECTION_SEED = (811, "unrolled_traj_global_projection")
UNROLLED_TRAJ_DAS_METHOD = (
    "das_ema_unrolled_trajectory_higher_noise_avg_shared_probe_projected4096_100t_mc10"
)
UNROLLED_TRAJ_DAS_CACHE_DIR = (
    ROOT / "query_variants" / "unrolled_trajectory_state_output_basis_vjp9_10q"
)
UNROLLED_TRAJ_DAS_SHARD_DIR = (
    ATTR_DIR / "_unrolled_trajectory_higher_noise_avg_shared_probe_10q_shards"
)


def unrolled_traj_das_shard_root(shard_index, shard_count):
    return UNROLLED_TRAJ_DAS_SHARD_DIR / (
        f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def lambda_tag(value):
    return str(float(value)).replace(".", "p")
