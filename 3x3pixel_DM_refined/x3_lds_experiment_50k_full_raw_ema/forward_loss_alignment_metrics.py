"""Shared score normalizations for learning and unlearning experiments."""

import numpy as np

from forward_loss_alignment_config import (
    FLA_LOG_EPS,
    FLA_LOSS_CONDITION_BINS,
    FLA_ROBUST_CLIP,
    FLA_ROBUST_SCALE_EPS,
)


def make_loss_condition_bins(baseline_events):
    order = np.argsort(baseline_events.mean(axis=1), kind="mergesort")
    return tuple(np.array_split(order, FLA_LOSS_CONDITION_BINS))


def loss_conditioned_robust_score(log_relative, condition_bins):
    """Robustly standardize within equal-count baseline-loss quantile bins."""
    result = np.empty_like(log_relative, dtype=np.float64)
    global_center = np.median(log_relative)
    global_mad = 1.4826 * np.median(np.abs(log_relative - global_center))
    scale_floor = max(FLA_ROBUST_SCALE_EPS, 0.05 * float(global_mad))
    for indices in condition_bins:
        values = log_relative[indices]
        center = np.median(values)
        scale = 1.4826 * np.median(np.abs(values - center))
        scale = max(float(scale), scale_floor)
        result[indices] = np.clip(
            (values - center) / scale,
            -FLA_ROBUST_CLIP,
            FLA_ROBUST_CLIP,
        )
    return result


def score_contributions(baseline_events, changed_events, condition_bins, direction):
    if direction == "decrease":
        numerator = baseline_events
        denominator = changed_events
        absolute = baseline_events - changed_events
    elif direction == "increase":
        numerator = changed_events
        denominator = baseline_events
        absolute = changed_events - baseline_events
    else:
        raise ValueError(f"Unknown score direction: {direction}")
    absolute = absolute.mean(axis=1)
    log_relative = np.log(
        (numerator + FLA_LOG_EPS) / (denominator + FLA_LOG_EPS)
    ).mean(axis=1)
    conditioned = loss_conditioned_robust_score(log_relative, condition_bins)
    return {
        "absolute": absolute,
        "log_relative": log_relative,
        "loss_conditioned_robust": conditioned,
    }
