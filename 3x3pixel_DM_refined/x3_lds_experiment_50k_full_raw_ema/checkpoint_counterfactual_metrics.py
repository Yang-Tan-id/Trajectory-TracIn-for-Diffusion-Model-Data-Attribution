"""Small metric helpers with no SciPy dependency."""

import numpy as np


def rankdata(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_values[1:] != sorted_values[:-1], True]
    )
    sorted_ranks = np.empty(values.shape[0], dtype=np.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sorted_ranks[start:end] = 0.5 * (start + end - 1) + 1.0
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def spearman_correlation(left, right):
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left_rank, right_rank) / denominator)
