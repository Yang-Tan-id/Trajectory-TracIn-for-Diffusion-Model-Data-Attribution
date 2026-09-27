"""Configuration for continuous reference learning from the first checkpoint."""

from checkpoint_adamw_reference_learning_config import *


C0RL_INITIAL_EPOCH = CARL_CHECKPOINT_EPOCHS[0]
C0RL_TARGET_FRACTION = 0.95
C0RL_MAX_STEPS = 200
C0RL_RESUME_EVERY = 5
C0RL_ROOT = ROOT / "checkpoint0_continuous_reference_learning"
C0RL_METHOD_RAW = (
    "checkpoint0_continuous_adamw_reference_learning_target0p95_"
    "max200_mc100_raw_decrease_q00_q09"
)
C0RL_METHOD_RATIO = (
    "checkpoint0_continuous_adamw_reference_learning_target0p95_"
    "max200_mc100_difference_over_sum_q00_q09"
)
C0RL_METHODS = (C0RL_METHOD_RAW, C0RL_METHOD_RATIO)
C0RL_LDS_METRICS = CARL_LDS_METRICS
