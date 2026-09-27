"""Configuration for checkpoint-AdamW reference-trajectory learning scores."""

from exp_config import *
from forward_loss_alignment_config import FLA_QUERY_DIR


CARL_QUERY_IDS = tuple(range(10))
CARL_FAMILY = "prompted"
CARL_CHECKPOINT_EPOCHS = tuple(
    range(BASE_SAVE_EVERY_EPOCHS, EPOCHS + 1, BASE_SAVE_EVERY_EPOCHS)
)
CARL_REFERENCE_STEPS = 4
CARL_REFERENCE_BATCH_SIZE = DDIM_STEPS // CARL_REFERENCE_STEPS
CARL_TRAIN_MC = 100
CARL_SCORE_DATAPOINT_BATCH = 128
CARL_EPS = 1e-12
CARL_MC_SEED_BASE = 946100
CARL_ROOT = ROOT / "checkpoint_adamw_reference_learning"
CARL_PARTIAL_DIR = CARL_ROOT / "partials"


def method_name(score_form, checkpoint_weighting):
    return (
        "checkpoint_adamw_reference_learning_4step_1000t_mc100_"
        f"{score_form}_{checkpoint_weighting}_q00_q09"
    )


CARL_METHOD_BY_VARIANT = {
    (score_form, weighting): method_name(score_form, weighting)
    for score_form in ("raw_decrease", "difference_over_sum")
    for weighting in ("uniform", "lr_weighted")
}
CARL_METHODS = tuple(CARL_METHOD_BY_VARIANT.values())

CARL_LDS_METRICS = (
    "endpoint_deviation_ema",
    "endpoint_deviation_raw",
    "trajectory_state_mse_ema",
    "trajectory_state_mse_raw",
)


def carl_checkpoint_path(epoch):
    return MODEL_DIR / "base" / CARL_FAMILY / f"epoch_{int(epoch):04d}.pt"
