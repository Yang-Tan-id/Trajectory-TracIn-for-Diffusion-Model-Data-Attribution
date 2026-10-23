"""Memory-bounded vectorized per-example gradient helpers."""

from __future__ import annotations

from typing import Iterator, Literal, Sequence

import torch
import torch.nn.functional as F


class LinearGradSampleCapture:
    """Exact per-sample gradients for a parameter set owned by Linear layers.

    A regular backward of ``losses.sum()`` supplies each Linear output's
    gradient.  Because Stable Diffusion has no cross-example attention, the
    weight gradient for example ``b`` is the outer-product sum
    ``grad_output[b].T @ input[b]``.  This avoids vectorizing backward itself.
    """

    def __init__(
        self,
        root: torch.nn.Module,
        parameters: Sequence[torch.nn.Parameter],
    ) -> None:
        self.parameters = list(parameters)
        self.parameter_slots = {
            id(parameter): slot for slot, parameter in enumerate(self.parameters)
        }
        owners: dict[int, tuple[torch.nn.Linear, str]] = {}
        relevant_modules: list[torch.nn.Linear] = []
        for module in root.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            owns_target = False
            for name, parameter in module.named_parameters(recurse=False):
                if id(parameter) in self.parameter_slots:
                    owners[id(parameter)] = (module, name)
                    owns_target = True
            if owns_target:
                relevant_modules.append(module)
        missing = [
            slot
            for slot, parameter in enumerate(self.parameters)
            if id(parameter) not in owners
        ]
        if missing:
            raise ValueError(
                "linear-hooks backend only supports parameters directly owned by "
                f"torch.nn.Linear; unmatched slots={missing[:8]}"
            )
        self.owners = owners
        self.enabled = False
        self.batch_size = 0
        self.samples: dict[int, torch.Tensor] = {}
        self.handles = [
            module.register_forward_hook(self._forward_hook)
            for module in relevant_modules
        ]

    def start(self, batch_size: int) -> None:
        if self.enabled:
            raise RuntimeError("linear gradient-sample capture is already active")
        if batch_size < 1:
            raise ValueError("linear gradient-sample batch size must be positive")
        self.batch_size = batch_size
        self.samples = {}
        self.enabled = True

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            raise TypeError("captured Linear must receive and return tensors")
        activation = inputs[0].detach()
        if activation.shape[0] != self.batch_size or output.shape[0] != self.batch_size:
            raise ValueError(
                "captured Linear does not preserve the leading example axis: "
                f"input={tuple(activation.shape)} output={tuple(output.shape)} "
                f"expected_batch={self.batch_size}"
            )

        def capture_output_gradient(gradient_output: torch.Tensor) -> None:
            input_flat = activation.float().reshape(self.batch_size, -1, activation.shape[-1])
            output_flat = gradient_output.detach().float().reshape(
                self.batch_size, -1, gradient_output.shape[-1]
            )
            if input_flat.shape[1] != output_flat.shape[1]:
                raise ValueError("captured Linear input/output token counts differ")
            for name, parameter in module.named_parameters(recurse=False):
                if id(parameter) not in self.parameter_slots:
                    continue
                if name == "weight":
                    value = torch.bmm(output_flat.transpose(1, 2), input_flat)
                elif name == "bias":
                    value = output_flat.sum(dim=1)
                else:
                    raise ValueError(f"unsupported Linear parameter {name!r}")
                previous = self.samples.get(id(parameter))
                self.samples[id(parameter)] = value if previous is None else previous + value

        output.register_hook(capture_output_gradient)

    def loss_gradients(
        self,
        losses: torch.Tensor,
        *,
        chunk_size: int,
    ) -> Iterator[tuple[list[int], list[torch.Tensor]]]:
        if not self.enabled:
            raise RuntimeError("call start() before the captured forward")
        if losses.ndim != 1 or losses.shape[0] != self.batch_size:
            raise ValueError("captured losses must have exactly one value per example")
        try:
            aggregate = torch.autograd.grad(
                losses.sum(), self.parameters, allow_unused=True
            )
        finally:
            self.enabled = False
        gradients = [
            self.samples.get(
                id(parameter),
                torch.zeros(
                    (self.batch_size,) + tuple(parameter.shape),
                    device=parameter.device,
                    dtype=torch.float32,
                ),
            )
            for parameter in self.parameters
        ]

        # The analytic per-sample gradients must add back to PyTorch's ordinary
        # aggregate gradient.  Fail closed if module ownership/shape assumptions
        # ever stop matching the installed Diffusers implementation.
        difference_square = torch.zeros((), device=losses.device)
        reference_square = torch.zeros((), device=losses.device)
        for parameter, per_sample, total in zip(
            self.parameters, gradients, aggregate
        ):
            expected = (
                torch.zeros_like(parameter) if total is None else total.detach()
            ).float()
            actual = per_sample.sum(dim=0).float()
            difference_square.add_((actual - expected).square().sum())
            reference_square.add_(expected.square().sum())
        relative_l2 = float(
            (difference_square.sqrt() / reference_square.sqrt().clamp_min(1e-30))
            .detach()
            .cpu()
        )
        if relative_l2 > 5e-3:
            raise RuntimeError(
                "linear-hooks per-sample gradients do not sum to the ordinary "
                f"gradient: relative_l2={relative_l2:.8e}"
            )

        for start in range(0, self.batch_size, chunk_size):
            end = min(start + chunk_size, self.batch_size)
            yield list(range(start, end)), [
                value[start:end].detach() for value in gradients
            ]


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
