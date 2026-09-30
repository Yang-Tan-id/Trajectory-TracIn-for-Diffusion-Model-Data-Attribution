"""Configuration for checkpoint-transition response diagnostics."""

from exp_config import *


CTD_QUERY_IDS = tuple(range(10))
CTD_PAIR_INDICES = tuple(range(49))
CTD_EVENTS_PER_INTERVAL = BASE_SAVE_EVERY_EPOCHS
CTD_ROOT = ROOT / "checkpoint_transition_diagnostic_49pair_10q_100t"
CTD_METHODS = (
    "exact_parameter_delta_start_jvp",
    "replayed_adamw_start_jvp",
    "current_bundle_raw_target_jvp",
    "current_interval_scaled_start_jvp",
    "current_interval_scaled_target_jvp",
)
CTD_QUERY_BATCH_SIZE = 256


def parse_integer_selection(value, default):
    if value is None:
        return list(default)
    result = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"invalid range: {token}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    if not result or len(result) != len(set(result)):
        raise ValueError("selection must be nonempty and contain no duplicates")
    return result
