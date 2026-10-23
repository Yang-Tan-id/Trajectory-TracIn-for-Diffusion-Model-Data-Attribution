"""Shared score contractions for ReTrac and endpoint methods."""

from __future__ import annotations

from typing import Mapping

import torch


EPS = 1e-12
TRANSFORMS = ("raw_gradient", "adamw_full")
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def normalize_rows(values: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def score_variants(query: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    """Return ``[variant, query, train]`` in the canonical variant order."""

    if query.ndim != 2 or train.ndim != 2:
        raise ValueError("query and train features must both be rank-2")
    if query.shape[1] != train.shape[1]:
        raise ValueError("query and train projection dimensions differ")
    query_normalized = normalize_rows(query)
    train_normalized = normalize_rows(train)
    return torch.stack(
        (
            query @ train.T,
            query_normalized @ train.T,
            query @ train_normalized.T,
            query_normalized @ train_normalized.T,
        ),
        dim=0,
    )


def dual_transform_scores(
    query: torch.Tensor,
    train_features: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return ``[transform, variant, query, train]``."""

    missing = set(TRANSFORMS).difference(train_features)
    if missing:
        raise KeyError(f"missing train transforms: {sorted(missing)}")
    return torch.stack(
        tuple(score_variants(query, train_features[name]) for name in TRANSFORMS),
        dim=0,
    )


def endpoint_pollute_reductions(components: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reduce ``[checkpoint, timestamp, ..., query, train]`` components."""

    if components.ndim < 4:
        raise ValueError("pollute components must include checkpoint and timestamp axes")
    linear = components.sum(dim=(0, 1))
    return {
        "linear": linear,
        "termwise_squared": components.square().sum(dim=(0, 1)),
        "timestamp_sum_squared": components.sum(dim=0).square().sum(dim=0),
        "total_sum_squared": linear.square(),
    }
