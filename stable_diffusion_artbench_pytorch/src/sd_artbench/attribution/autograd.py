"""Memory-bounded vectorized per-example gradient helpers."""

from __future__ import annotations

from typing import Iterator, Literal, Sequence

import torch
import torch.nn.functional as F


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
