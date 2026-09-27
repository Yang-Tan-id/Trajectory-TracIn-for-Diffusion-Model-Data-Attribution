"""Configuration for the five-stage 50k / balanced-50%-LDS experiment."""

from pathlib import Path

from exp_config import *


STAGED_ROOT = Path("x3_staged_lds_exp_50k")
STAGED_PARTITION_DIR = STAGED_ROOT / "partition"
STAGED_MASK_DIR = STAGED_ROOT / "subset_masks"
STAGED_MODEL_DIR = STAGED_ROOT / "models"
STAGED_QUERY_DIR = STAGED_ROOT / "queries"
STAGED_ATTR_DIR = STAGED_ROOT / "attribution"
STAGED_LDS_DIR = STAGED_ROOT / "lds"
STAGED_LOG_DIR = STAGED_ROOT / "logs"

STAGE_COUNT = 5
STAGE_SIZE = 10_000
STAGE_EPOCHS = 40
STAGED_EPOCHS = STAGE_COUNT * STAGE_EPOCHS
STAGED_SAVE_EVERY = 4
STAGED_CHECKPOINT_COUNT = STAGED_EPOCHS // STAGED_SAVE_EVERY
STAGED_LDS_MASK_COUNT = 192
STAGED_LDS_PER_STAGE = STAGE_SIZE // 2
STAGED_LDS_SUBSET_SIZE = STAGE_COUNT * STAGED_LDS_PER_STAGE
STAGED_PARTITION_SEED = DATA_SEED
STAGED_MASK_SEED = 67050
STAGED_QUERY_IDS = tuple(range(10))
STAGED_FAMILY = "prompted"
STAGED_TRAJ_METHOD = "staged_next_projected_first_raw_timestamp_sum_squared"
STAGED_DAS_METHOD = "staged_das_final_ema_full50k"


def staged_base_checkpoint(epoch):
    return STAGED_MODEL_DIR / "base" / f"epoch_{int(epoch):04d}.pt"


def staged_subset_checkpoint(mask_id):
    return (
        STAGED_MODEL_DIR
        / "subsets"
        / f"subset_{int(mask_id):03d}"
        / f"epoch_{STAGED_EPOCHS:04d}.pt"
    )


def stage_for_interval(source_epoch, target_epoch):
    if int(target_epoch) != int(source_epoch) + STAGED_SAVE_EVERY:
        raise ValueError((source_epoch, target_epoch))
    return min((int(target_epoch) - 1) // STAGE_EPOCHS, STAGE_COUNT - 1)
