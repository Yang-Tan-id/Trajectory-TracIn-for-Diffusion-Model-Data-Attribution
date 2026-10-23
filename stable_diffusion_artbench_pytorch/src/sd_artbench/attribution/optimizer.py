"""Reconstruct per-example AdamW update directions from saved optimizer state."""

from __future__ import annotations

import math
from typing import Sequence

import torch


@torch.no_grad()
def adamw_full_direction(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float | None = 1.0,
) -> tuple[list[torch.Tensor], float]:
    """Return the next AdamW direction with learning rate factored out.

    Historical first/second moments are read but never mutated. The candidate
    gradient is globally clipped like training when ``max_grad_norm`` is set.
    Returned tensors equal ``delta_parameter / lr`` (including the minus sign
    and decoupled weight decay), allowing TracIn to apply checkpoint LR once.
    """

    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have the same length")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")
    if max_grad_norm is None:
        clip_scale = 1.0
    else:
        global_norm = torch.sqrt(
            sum(gradient.detach().float().square().sum() for gradient in gradients)
        )
        clip_scale = min(1.0, max_grad_norm / max(float(global_norm.cpu()), 1e-12))
    group_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    directions: list[torch.Tensor] = []
    learning_rates = set()
    for parameter, gradient in zip(parameters, gradients):
        group = group_by_parameter.get(id(parameter))
        if group is None:
            raise KeyError("parameter is absent from optimizer param_groups")
        if group.get("amsgrad", False):
            raise NotImplementedError("AMSGrad is not supported by this attribution path")
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        learning_rates.add(float(group["lr"]))
        state = optimizer.state.get(parameter, {})
        step_value = state.get("step", 0)
        step = int(step_value.item() if torch.is_tensor(step_value) else step_value) + 1
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        if exp_avg_sq is None:
            exp_avg_sq = torch.zeros_like(parameter, memory_format=torch.preserve_format)

        grad = gradient.detach().to(dtype=parameter.dtype).mul(clip_scale)
        next_avg = exp_avg.detach().mul(beta1).add(grad, alpha=1.0 - beta1)
        next_avg_sq = exp_avg_sq.detach().mul(beta2).addcmul(
            grad, grad, value=1.0 - beta2
        )
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denominator = next_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add(eps)
        adaptive = next_avg.div(bias_correction1).div(denominator)
        directions.append(adaptive.add(parameter.detach(), alpha=weight_decay).neg())

    if len(learning_rates) != 1:
        raise ValueError(
            "attribution currently requires one shared LR across AdamW param groups"
        )
    return directions, learning_rates.pop()
