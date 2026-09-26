"""Configuration for checkpoint/timestamp counterfactual experiments."""

from exp_config import ROOT, TRACIN_PROJECTED_BATCH_SIZE, TRACIN_TRAIN_MC


CF_ROOT = ROOT / "checkpoint_timestamp_counterfactual"
CF_UPDATE_ROOT = CF_ROOT / "subset_unlearning_updates"
CF_RESPONSE_ROOT = CF_ROOT / "subset_unlearning_responses"
CF_DIRECTION_SCORE_METHOD = "last_noise_delta_direction_raw_linear"

CF_TRAIN_MC = int(TRACIN_TRAIN_MC)
CF_UPDATE_GRAD_BATCH_SIZE = int(TRACIN_PROJECTED_BATCH_SIZE)
# Delta-direction scoring materializes one per-example gradient matrix but the
# 3x3 model/input is small. A larger batch removes thousands of tiny GPU
# launches per checkpoint/timestamp term without changing score semantics.
CF_DIRECTION_GRAD_BATCH_SIZE = 640
CF_UPDATE_SCALE = 1.0
CF_FINAL_PARAM_SOURCE = "ema"
CF_UNLEARN_SET = "removed"
CF_DELTA_NORMALIZE = True
