"""Configuration for endpoint-MC100 AdamW unlearning scores."""

from exp_config import *


MUCS_QUERY_IDS = tuple(range(10))
MUCS_FAMILY = "prompted"
MUCS_NULL_EPOCH = BASE_SAVE_EVERY_EPOCHS
MUCS_INITIAL_EPOCH = EPOCHS
MUCS_ENDPOINT_MC = 100
MUCS_TRAIN_MC = 100
MUCS_LR_SCALE = 0.1
MUCS_UNLEARNING_LR = PEAK_LR * MUCS_LR_SCALE
MUCS_NULL_GAP_FRACTION = 0.95
MUCS_EPS = 1e-12
MUCS_MAX_UNLEARNING_STEPS = 5000
MUCS_RESUME_EVERY_STEPS = 10
MUCS_SCORE_DATAPOINT_BATCH = 128
MUCS_ENDPOINT_MC_SEED_BASE = 927000
MUCS_TRAIN_MC_SEED = 927100
MUCS_FT_SHUFFLE_SEED_BASE = 927200
MUCS_FT_DIFFUSION_SEED_BASE = 927300
MUCS_LAMBDA = 1.0
MUCS_FT_BATCH_SIZE = BATCH_SIZE

# Keep the original, query-ascent-only name reserved so old results cannot be
# mistaken for the true joint MUCS objective.
MUCS_GA_ONLY_METHOD = "mucs_endpoint_mc100_adamw_lr0p1_nullgap95_raw_q00_q09"
MUCS_METHOD = (
    "mucs_joint_ft_ga_endpoint_mc100_adamw_lambda1_"
    "lr0p1_nullgap95_raw_q00_q09"
)
MUCS_ROOT = ROOT / "mucs_endpoint_unlearning" / MUCS_METHOD

MUCS_LDS_METRICS = (
    "endpoint_deviation_ema",
    "endpoint_deviation_raw",
    "trajectory_state_mse_ema",
    "trajectory_state_mse_raw",
)


def mucs_checkpoint_path(epoch):
    return MODEL_DIR / "base" / MUCS_FAMILY / f"epoch_{int(epoch):04d}.pt"
