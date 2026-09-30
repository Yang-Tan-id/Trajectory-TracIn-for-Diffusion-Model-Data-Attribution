"""Configuration for two-checkpoint frozen-gradient AdamW diagnostics."""

from checkpoint_transition_diagnostic_config import *


CEAD_ROOT = ROOT / "checkpoint_endpoint_adam_diagnostic_10q_100t"
CEAD_METHODS = (
    "exact_parameter_delta_start_jvp",
    "frozen_start_gradient_adamw_jvp",
    "frozen_target_gradient_adamw_jvp",
    "frozen_endpoint_trapezoid_adamw_jvp",
)


def cead_root(loss_mc):
    if int(loss_mc) == 1:
        return CEAD_ROOT
    return ROOT / f"checkpoint_endpoint_adam_diagnostic_mc{int(loss_mc)}_10q_100t"
