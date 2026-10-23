"""Memory-bounded vectorized per-example gradient helpers."""

from __future__ import annotations

from typing import Iterator, Sequence

import torch


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
