"""Configuration for timestamp-aligned SOURCE-DAS on the trained X3 models."""

import math
import os
from pathlib import Path

from exp_config import *


_DEFAULT_SIMPLE_INFLUENCE = Path(__file__).resolve().parents[3] / "simple-influence"
SOURCE_DAS_SIMPLE_INFLUENCE_ROOT = Path(
    os.environ.get("SIMPLE_INFLUENCE_ROOT", str(_DEFAULT_SIMPLE_INFLUENCE))
)

SOURCE_DAS_METHOD = "source_das_raw_to_ema_10ckpt_100t_mc10_exact"
SOURCE_DAS_ROOT = ROOT / "source_das"
SOURCE_DAS_SHARD_ROOT = SOURCE_DAS_ROOT / "timestamp_shards"

SOURCE_DAS_CHECKPOINT_EPOCHS = tuple(range(20, EPOCHS + 1, 20))
SOURCE_DAS_NUM_SEGMENTS = len(SOURCE_DAS_CHECKPOINT_EPOCHS)
SOURCE_DAS_TRAIN_MC = 10
SOURCE_DAS_TRAIN_SCORE_BATCH_SIZE = 64
SOURCE_DAS_FINAL_PARAM_SOURCE = "ema"
SOURCE_DAS_CHECKPOINT_PARAM_SOURCE = "raw"
SOURCE_DAS_USE_TRUE_FISHER = False
SOURCE_DAS_OUTPUT_DIM = 3 * 3 * 3

SOURCE_DAS_INFLUENCE_MODULES = (
    "time_mlp.0",
    "time_mlp.2",
    "cond_mlp.0",
    "cond_mlp.2",
    "in_conv",
    "block1.2",
    "block2.2",
    "block3.2",
    "out_conv",
    "emb_to_bias",
)


def source_das_steps_per_epoch():
    return math.ceil(N_TRAIN / BATCH_SIZE)


def source_das_segment_boundaries():
    return (0,) + SOURCE_DAS_CHECKPOINT_EPOCHS


def source_das_iters_per_segment():
    steps = source_das_steps_per_epoch()
    boundaries = source_das_segment_boundaries()
    return tuple(
        (boundaries[index + 1] - boundaries[index]) * steps
        for index in range(SOURCE_DAS_NUM_SEGMENTS)
    )


def _lr_at(step, total_steps):
    warm = int(math.ceil(total_steps * WARMUP_RATIO))
    if warm > 0 and step < warm:
        return PEAK_LR * float(step + 1) / float(warm)
    if total_steps <= warm:
        return PEAK_LR
    progress = (step - warm) / max(1, total_steps - warm)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * PEAK_LR * (1.0 + math.cos(math.pi * progress))


def source_das_lrs_per_segment():
    steps_per_epoch = source_das_steps_per_epoch()
    total_steps = EPOCHS * steps_per_epoch
    boundaries = source_das_segment_boundaries()
    values = []
    for index in range(SOURCE_DAS_NUM_SEGMENTS):
        start = boundaries[index] * steps_per_epoch
        end = boundaries[index + 1] * steps_per_epoch
        values.append(
            sum(_lr_at(step, total_steps) for step in range(start, end))
            / float(end - start)
        )
    return tuple(values)
