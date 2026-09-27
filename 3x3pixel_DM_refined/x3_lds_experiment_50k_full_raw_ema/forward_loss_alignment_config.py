"""Configuration for the checkpoint forward-loss-alignment experiment."""

from exp_config import *


FLA_ROOT = ROOT / "forward_loss_alignment"
FLA_REPLAY_DIR = FLA_ROOT / "training_event_replay"
FLA_QUERY_DIR = FLA_ROOT / "reference_queries"
FLA_BASELINE_DIR = FLA_ROOT / "baseline_losses"
FLA_PARTIAL_DIR = FLA_ROOT / "partials"

# The base model was saved every four epochs, so each checkpoint interval has
# exactly four realized training events for every datapoint.
FLA_CHECKPOINT_EPOCHS = tuple(range(BASE_SAVE_EVERY_EPOCHS, EPOCHS + 1, BASE_SAVE_EVERY_EPOCHS))
FLA_EVENTS_PER_CHECKPOINT = BASE_SAVE_EVERY_EPOCHS
FLA_QUERY_IDS = tuple(range(50))
FLA_FAMILY = "prompted"
FLA_REFERENCE_PARAM_SOURCE = "ema"
FLA_CHECKPOINT_PARAM_SOURCE = "raw"

FLA_METHOD_RAW_STEP = "forward_loss_alignment_raw_sgd_50ckpt_1000t_4event"
FLA_METHOD_NORMALIZED_STEP = "forward_loss_alignment_normalized_sgd_50ckpt_1000t_4event"
FLA_METHODS = (FLA_METHOD_RAW_STEP, FLA_METHOD_NORMALIZED_STEP)

# This is a datapoint batch. Four realized training events are flattened into
# a forward batch of 4 * FLA_DATAPOINT_BATCH_SIZE.
FLA_DATAPOINT_BATCH_SIZE = 640
FLA_QUERY_BATCH_SIZE = 250
FLA_NORMALIZE_EPS = 1e-12

FLA_LDS_METRICS = (
    "endpoint_deviation_ema",
    "endpoint_deviation_raw",
    "trajectory_state_mse_ema",
    "trajectory_state_mse_raw",
)


def checkpoint_path(epoch):
    return MODEL_DIR / "base" / FLA_FAMILY / f"epoch_{int(epoch):04d}.pt"


def replay_t_path():
    return FLA_REPLAY_DIR / "t.npy"


def replay_noise_path():
    return FLA_REPLAY_DIR / "noise.npy"


def baseline_path():
    return FLA_BASELINE_DIR / f"{FLA_FAMILY}_raw.npy"
