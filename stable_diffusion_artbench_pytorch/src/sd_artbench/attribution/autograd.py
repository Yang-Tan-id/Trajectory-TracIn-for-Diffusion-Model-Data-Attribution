"""Memory-bounded vectorized per-example gradient helpers."""

from __future__ import annotations

from typing import Iterator, Literal, Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def compare_gradient_batches(
    reference: Sequence[torch.Tensor],
    candidate: Sequence[torch.Tensor],
) -> list[dict[str, float]]:
    """Return whole-gradient diagnostics for each example in two gradient trees."""

    if len(reference) != len(candidate) or not reference:
        raise ValueError("gradient trees must have the same non-zero leaf count")
    batch_size = reference[0].shape[0]
    if any(
        left.shape != right.shape or left.shape[0] != batch_size
        for left, right in zip(reference, candidate)
    ):
        raise ValueError("gradient trees must have identical batched leaf shapes")
    left_square = torch.zeros(batch_size, device=reference[0].device)
    right_square = torch.zeros_like(left_square)
    difference_square = torch.zeros_like(left_square)
    dot = torch.zeros_like(left_square)
    for left, right in zip(reference, candidate):
        left_flat = left.detach().float().reshape(batch_size, -1)
        right_flat = right.detach().float().reshape(batch_size, -1)
        left_square.add_(left_flat.square().sum(dim=1))
        right_square.add_(right_flat.square().sum(dim=1))
        difference_square.add_((right_flat - left_flat).square().sum(dim=1))
        dot.add_((left_flat * right_flat).sum(dim=1))
    left_norm = left_square.sqrt().clamp_min(1e-30)
    right_norm = right_square.sqrt().clamp_min(1e-30)
    relative_l2 = difference_square.sqrt() / left_norm
    cosine = dot / (left_norm * right_norm).clamp_min(1e-30)
    norm_ratio = right_norm / left_norm
    return [
        {
            "relative_l2": float(relative_l2[index].cpu()),
            "cosine": float(cosine[index].cpu()),
            "norm_ratio": float(norm_ratio[index].cpu()),
        }
        for index in range(batch_size)
    ]


def batched_loss_gradients(
    losses: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    sample_indices: Sequence[int],
    *,
    chunk_size: int,
) -> Iterator[tuple[list[int], list[torch.Tensor]]]:
    """Yield vectorized per-example VJPs in memory-bounded chunks.

    PyTorch's ``is_grads_batched`` uses its vmap backend to evaluate several
    one-hot VJPs together without functionalizing the Diffusers U-Net.
    """

    if losses.ndim != 1:
        raise ValueError("losses must be a rank-1 per-example vector")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    indices = [int(index) for index in sample_indices]
    for start in range(0, len(indices), chunk_size):
        current = indices[start : start + chunk_size]
        rows = torch.as_tensor(current, device=losses.device, dtype=torch.long)
        basis = torch.nn.functional.one_hot(rows, num_classes=losses.shape[0]).to(
            dtype=losses.dtype
        )
        values = torch.autograd.grad(
            losses,
            parameters,
            grad_outputs=basis,
            allow_unused=True,
            retain_graph=start + len(current) < len(indices),
            is_grads_batched=True,
        )
        gradients_batch = [
            torch.zeros(
                (len(current),) + tuple(parameter.shape),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            if value is None
            else value.detach()
            for parameter, value in zip(parameters, values)
        ]
        yield current, gradients_batch


def scalar_loop_loss_gradients(
    losses: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    sample_indices: Sequence[int],
    *,
    chunk_size: int,
) -> Iterator[tuple[list[int], list[torch.Tensor]]]:
    """Yield exact per-example gradients using ordinary scalar VJPs.

    All examples share the original batched forward graph.  Backward is run
    once per scalar loss without ``is_grads_batched`` or ``torch.func``.  This
    is the correctness backend for operator stacks whose vmap batching rules
    do not reproduce scalar autograd.
    """

    if losses.ndim != 1:
        raise ValueError("losses must be a rank-1 per-example vector")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    indices = [int(index) for index in sample_indices]
    for start in range(0, len(indices), chunk_size):
        current = indices[start : start + chunk_size]
        per_example: list[list[torch.Tensor]] = []
        for index in current:
            values = torch.autograd.grad(
                losses[index],
                parameters,
                allow_unused=True,
                retain_graph=index != indices[-1],
            )
            per_example.append(
                [
                    torch.zeros_like(parameter) if value is None else value.detach()
                    for parameter, value in zip(parameters, values)
                ]
            )
        gradients_batch = [
            torch.stack([values[parameter_slot] for values in per_example], dim=0)
            for parameter_slot in range(len(parameters))
        ]
        yield current, gradients_batch


def _functional_parameter_dict(
    module: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Map an externally ordered parameter list to names in ``module``.

    The LoRA parameters are registered by both ``AttnProcsLayers`` and the
    U-Net.  ``torch.func.functional_call`` needs the U-Net names, while all
    downstream optimizer/projection code needs the original optimizer order.
    """

    name_by_id = {id(parameter): name for name, parameter in module.named_parameters()}
    names = []
    missing = []
    for slot, parameter in enumerate(parameters):
        name = name_by_id.get(id(parameter))
        if name is None:
            missing.append(slot)
        else:
            names.append(name)
    if missing:
        raise ValueError(
            "functional per-example gradients could not find U-Net names for "
            f"parameter slots {missing[:8]}"
        )
    if len(set(names)) != len(names):
        raise ValueError("functional per-example gradient parameter names are not unique")
    return names, {
        name: parameter.detach() for name, parameter in zip(names, parameters)
    }


def functional_per_example_gradients(
    module: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss: Literal["mse", "pollute"],
    chunk_size: int,
) -> Iterator[tuple[list[int], list[torch.Tensor]]]:
    """Yield per-example gradients using the public ``torch.func`` path.

    Unlike ``is_grads_batched=True``, this applies ``vmap(grad(loss_fn))`` to a
    functional U-Net.  That avoids the legacy batched-backward fallbacks used
    by PyTorch 2.0 for LayerNorm, softmax, upsampling, and MSE backward.

    For ``loss='mse'``, ``targets`` contains epsilon/v-prediction targets.  For
    ``loss='pollute'``, it contains a detached predicted-noise delta.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    batch_size = noisy_latents.shape[0]
    if not (
        timesteps.shape[0]
        == encoder_hidden_states.shape[0]
        == targets.shape[0]
        == batch_size
    ):
        raise ValueError("all functional-gradient inputs must share a batch axis")
    if loss not in {"mse", "pollute"}:
        raise ValueError(f"unsupported functional loss: {loss}")

    names, functional_parameters = _functional_parameter_dict(module, parameters)

    def single_loss(
        current_parameters: dict[str, torch.Tensor],
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        hidden: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = torch.func.functional_call(
            module,
            current_parameters,
            (
                noisy_latent.unsqueeze(0),
                timestep.unsqueeze(0),
                hidden.unsqueeze(0),
            ),
            strict=False,
        ).sample
        if loss == "mse":
            return F.mse_loss(
                prediction.float(), target.unsqueeze(0).float(), reduction="mean"
            )
        return (prediction.float() * target.unsqueeze(0).float()).mean()

    gradient_function = torch.func.grad(single_loss)
    vectorized_gradient = torch.func.vmap(
        gradient_function,
        in_dims=(None, 0, 0, 0, 0),
        randomness="error",
    )
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        values = vectorized_gradient(
            functional_parameters,
            noisy_latents[start:end],
            timesteps[start:end],
            encoder_hidden_states[start:end],
            targets[start:end],
        )
        yield list(range(start, end)), [values[name].detach() for name in names]
