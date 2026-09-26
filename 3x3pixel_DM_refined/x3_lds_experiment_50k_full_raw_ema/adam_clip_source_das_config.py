"""Configuration for raw-parameter Adam/clipping-aware SOURCE-DAS."""

from exp_config import *
from source_das_config import SOURCE_DAS_INFLUENCE_MODULES


ADAM_CLIP_SOURCE_ROOT = ROOT / "source_das_adam_clip_raw_11h50p_lrweighted"
ADAM_CLIP_SOURCE_SHARD_ROOT = ADAM_CLIP_SOURCE_ROOT / "timestamp_shards"

# Ten 20-epoch dynamics segments. Curvature and datapoint gradients use one
# midpoint checkpoint per segment, plus the final epoch-200 endpoint in the
# last segment: ten midpoints + one final endpoint = eleven checkpoints total.
ADAM_CLIP_SOURCE_SEGMENT_BOUNDARIES = tuple(range(0, EPOCHS + 1, 20))
ADAM_CLIP_SOURCE_CURVATURE_EPOCHS_PER_SEGMENT = tuple(
    ((start + 12,) if end < EPOCHS else (start + 12, end))
    for start, end in zip(
        ADAM_CLIP_SOURCE_SEGMENT_BOUNDARIES[:-1],
        ADAM_CLIP_SOURCE_SEGMENT_BOUNDARIES[1:],
    )
)
ADAM_CLIP_SOURCE_NUM_SEGMENTS = len(
    ADAM_CLIP_SOURCE_CURVATURE_EPOCHS_PER_SEGMENT
)

# Adam/clipping dynamics use every saved checkpoint in each segment.  A
# checkpoint at epoch e represents the preceding four-epoch interval.
ADAM_CLIP_SOURCE_P_CHECKPOINT_EPOCHS = tuple(
    tuple(range(start + 4, end + 1, 4))
    for start, end in zip(
        ADAM_CLIP_SOURCE_SEGMENT_BOUNDARIES[:-1],
        ADAM_CLIP_SOURCE_SEGMENT_BOUNDARIES[1:],
    )
)
ADAM_CLIP_SOURCE_TRAIN_MC = 10
ADAM_CLIP_SOURCE_TRAIN_BATCH_SIZE = 2 * BATCH_SIZE
ADAM_CLIP_SOURCE_CLIP_REPLAY_BATCH_SIZE = BATCH_SIZE
ADAM_CLIP_SOURCE_FINAL_PARAM_SOURCE = "raw"
ADAM_CLIP_SOURCE_CHECKPOINT_PARAM_SOURCE = "raw"
ADAM_CLIP_SOURCE_OUTPUT_DIM = 3 * 3 * 3
ADAM_CLIP_SOURCE_INFLUENCE_MODULES = SOURCE_DAS_INFLUENCE_MODULES

# The original training clips the global batch-gradient norm to this value.
ADAM_CLIP_SOURCE_CLIP_NORM = GRAD_CLIP

# H^{-1} is evaluated in the EK-FAC basis.  The relative floor implements a
# stable pseudoinverse for numerically zero empirical-Fisher eigenvalues.
ADAM_CLIP_SOURCE_EIGENVALUE_RELATIVE_FLOOR = 1e-6
ADAM_CLIP_SOURCE_EIGENVALUE_ABSOLUTE_FLOOR = 1e-12
ADAM_CLIP_SOURCE_NORM_EPS = 1e-12

ADAM_CLIP_SOURCE_METHODS = {
    "unnormalized": "source_das_adam_clip_raw_11h50p_100t_mc10_unnormalized",
    "jacobian_fro_rms": "source_das_adam_clip_raw_11h50p_100t_mc10_jacobian_fro_rms",
}


def adam_clip_source_steps_per_epoch():
    return (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE


def adam_clip_source_segment_boundaries():
    return ADAM_CLIP_SOURCE_SEGMENT_BOUNDARIES


def adam_clip_source_iters_per_segment():
    steps = adam_clip_source_steps_per_epoch()
    boundaries = adam_clip_source_segment_boundaries()
    return tuple(
        (boundaries[index + 1] - boundaries[index]) * steps
        for index in range(ADAM_CLIP_SOURCE_NUM_SEGMENTS)
    )


def _lr_at(step, total_steps):
    import math

    warm = int(math.ceil(total_steps * WARMUP_RATIO))
    if warm > 0 and step < warm:
        return PEAK_LR * float(step + 1) / float(warm)
    if total_steps <= warm:
        return PEAK_LR
    progress = (step - warm) / max(1, total_steps - warm)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * PEAK_LR * (1.0 + math.cos(math.pi * progress))


def adam_clip_source_lr_sums_per_segment():
    steps_per_epoch = adam_clip_source_steps_per_epoch()
    total_steps = EPOCHS * steps_per_epoch
    boundaries = adam_clip_source_segment_boundaries()
    return tuple(
        sum(
            _lr_at(step, total_steps)
            for step in range(
                boundaries[index] * steps_per_epoch,
                boundaries[index + 1] * steps_per_epoch,
            )
        )
        for index in range(ADAM_CLIP_SOURCE_NUM_SEGMENTS)
    )


def adam_clip_source_p_lr_weights_per_segment():
    """LR integrals for the five four-epoch p/c samples in each segment."""
    steps_per_epoch = adam_clip_source_steps_per_epoch()
    total_steps = EPOCHS * steps_per_epoch
    result = []
    for epochs in ADAM_CLIP_SOURCE_P_CHECKPOINT_EPOCHS:
        weights = []
        previous_epoch = epochs[0] - 4
        for epoch in epochs:
            weights.append(
                sum(
                    _lr_at(step, total_steps)
                    for step in range(
                        previous_epoch * steps_per_epoch,
                        epoch * steps_per_epoch,
                    )
                )
            )
            previous_epoch = epoch
        result.append(tuple(weights))
    return tuple(result)
