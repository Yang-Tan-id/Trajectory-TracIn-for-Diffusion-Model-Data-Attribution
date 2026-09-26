"""Adam-preconditioned, clipping-aware SOURCE for the X3 experiment.

The clipping Jacobian uses a frozen-scale approximation.  For each replayed
batch we estimate c=min(1, C/||g_batch||), average c within a checkpoint and
segment, and use c in the segment decay S.  The derivative of c itself cannot
be recovered because the original shuffled batches and their gradients were
not saved.
"""

import torch

from adam_clip_source_das_config import *
from source_das_x3 import X3SourceComputer

from src.ekfac_utils import make_grads_dict_to_matrix


def _optimizer_state_by_parameter_name(payload, model):
    optimizer = payload["optimizer_state"]
    parameter_ids = []
    for group in optimizer["param_groups"]:
        parameter_ids.extend(group["params"])
    named_parameters = list(model.named_parameters())
    if len(parameter_ids) != len(named_parameters):
        raise ValueError(
            "optimizer/model parameter count mismatch: "
            f"{len(parameter_ids)} != {len(named_parameters)}"
        )
    state = optimizer["state"]
    result = {}
    for (name, parameter), parameter_id in zip(named_parameters, parameter_ids):
        item = state[parameter_id]
        if "exp_avg_sq" not in item:
            raise ValueError(f"optimizer state for {name} has no exp_avg_sq")
        if tuple(item["exp_avg_sq"].shape) != tuple(parameter.shape):
            raise ValueError(f"optimizer state shape mismatch for {name}")
        result[name] = item
    return result


def _bias_corrected_adam_preconditioner(item, beta2, eps, device):
    step = item.get("step", 0)
    if torch.is_tensor(step):
        step = int(step.item())
    else:
        step = int(step)
    if step <= 0:
        raise ValueError(f"invalid Adam step {step}")
    correction = 1.0 - float(beta2) ** step
    second_moment = item["exp_avg_sq"].to(device=device, dtype=torch.float32)
    second_moment = second_moment / correction
    return torch.reciprocal(torch.sqrt(second_moment) + float(eps))


def _module_parameter_matrix(module, module_name, tensors):
    value = tensors[module_name + ".weight"]
    if isinstance(module, torch.nn.Conv2d):
        value = value.reshape(value.shape[0], -1)
    if module.bias is not None:
        value = torch.cat((value, tensors[module_name + ".bias"].unsqueeze(-1)), -1)
    return value


class X3AdamClipSourceComputer(X3SourceComputer):
    """SOURCE with Adam diagonal P, diagonal S, and query-only normalization."""

    def __init__(
        self,
        *args,
        preconditioner_checkpoints_per_segment,
        preconditioner_lr_weights_per_segment,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(preconditioner_checkpoints_per_segment) != len(
            self.checkpoints_per_segment
        ):
            raise ValueError("preconditioner checkpoint segment count mismatch")
        if len(preconditioner_lr_weights_per_segment) != len(
            preconditioner_checkpoints_per_segment
        ):
            raise ValueError("preconditioner LR-weight segment count mismatch")
        for checkpoints, weights in zip(
            preconditioner_checkpoints_per_segment,
            preconditioner_lr_weights_per_segment,
        ):
            if len(checkpoints) != len(weights) or not checkpoints:
                raise ValueError("each p/c checkpoint needs one positive LR weight")
            if any(float(weight) <= 0.0 for weight in weights):
                raise ValueError("p/c LR weights must be positive")
        self.preconditioner_checkpoints_per_segment = (
            preconditioner_checkpoints_per_segment
        )
        self.preconditioner_lr_weights_per_segment = (
            preconditioner_lr_weights_per_segment
        )

    def _checkpoint_preconditioner(self, checkpoint, segment):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._load_checkpoint(checkpoint)
        optimizer_state = _optimizer_state_by_parameter_name(payload, self.model)
        config = payload.get("config", {})
        beta2 = float(config.get("adam_b2", ADAM_B2))
        adam_eps = float(config.get("adam_eps", ADAM_EPS))
        tensors = {
            name: _bias_corrected_adam_preconditioner(
                optimizer_state[name], beta2, adam_eps, self.task.device
            )
            for name, _ in self.model.named_parameters()
        }
        return {
            name: _module_parameter_matrix(module, name, tensors).detach()
            for name, module in zip(segment.modules_name, segment.modules)
        }

    def _clip_scale_from_batch_grads(self, grads_dict, batch_size):
        norm_sq = torch.zeros((), device=self.task.device)
        with torch.no_grad():
            for value in grads_dict.values():
                # Per-example vmap gradients have a leading batch axis.  A
                # regular batch gradient is the gradient of the summed loss.
                if value.shape[0] == batch_size:
                    batch_gradient = value.mean(dim=0)
                else:
                    batch_gradient = value / float(batch_size)
                norm_sq.add_(batch_gradient.float().square().sum())
        gradient_norm = float(torch.sqrt(norm_sq).item())
        return min(
            1.0,
            float(ADAM_CLIP_SOURCE_CLIP_NORM)
            / max(gradient_norm, ADAM_CLIP_SOURCE_NORM_EPS),
        )

    def _estimate_clip_scale(self, checkpoint, loader):
        """Estimate c with one ordinary batch-gradient pass (no per-example vmap)."""
        self._load_checkpoint(checkpoint)
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        grad_fn = self._compute_train_loss_grad()
        examples_seen = 0
        scale_sum = 0.0
        clipped_examples = 0
        for batch in loader:
            batch_size = self.task.get_batch_size(batch)
            grads_dict = grad_fn(params, buffers, batch)
            # grad_fn differentiates the sum of per-example losses.
            norm_sq = torch.zeros((), device=self.task.device)
            with torch.no_grad():
                for value in grads_dict.values():
                    norm_sq.add_((value.float() / float(batch_size)).square().sum())
            gradient_norm = float(torch.sqrt(norm_sq).item())
            clip_scale = min(
                1.0,
                float(ADAM_CLIP_SOURCE_CLIP_NORM)
                / max(gradient_norm, ADAM_CLIP_SOURCE_NORM_EPS),
            )
            scale_sum += clip_scale * batch_size
            if clip_scale < 1.0:
                clipped_examples += batch_size
            examples_seen += batch_size
            del grads_dict
        if examples_seen != len(loader.dataset):
            raise RuntimeError(
                f"clip pass saw {examples_seen}, expected {len(loader.dataset)}"
            )
        return (
            scale_sum / float(examples_seen),
            clipped_examples / float(examples_seen),
        )

    def _diagonal_and_clip(self, checkpoint, segment, loader):
        """Compute direct diagonal empirical Fisher and c at one checkpoint."""
        self._load_checkpoint(checkpoint)
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        diagonal_sums = {
            name: torch.zeros_like(segment.kronecker_eigvals[name])
            for name in segment.modules_name
        }
        examples_seen = 0
        scale_sum = 0.0
        clipped_examples = 0
        for batch in loader:
            batch_size = self.task.get_batch_size(batch)
            grads_dict = self._train_loss_grads_dict(batch, params, buffers)
            clip_scale = self._clip_scale_from_batch_grads(grads_dict, batch_size)
            scale_sum += clip_scale * batch_size
            if clip_scale < 1.0:
                clipped_examples += batch_size
            with torch.no_grad():
                for name, module in zip(segment.modules_name, segment.modules):
                    matrix = make_grads_dict_to_matrix(
                        module=module,
                        module_name=name,
                        grads_dict=grads_dict,
                        remove_grads=False,
                    ).to(dtype=self.grads_dtype)
                    diagonal_sums[name].add_(matrix.square().sum(dim=0))
            examples_seen += batch_size
            del grads_dict
        if examples_seen != len(loader.dataset):
            raise RuntimeError(
                f"diagonal pass saw {examples_seen}, expected {len(loader.dataset)}"
            )
        return (
            {
                name: value / float(examples_seen)
                for name, value in diagonal_sums.items()
            },
            scale_sum / float(examples_seen),
            clipped_examples / float(examples_seen),
        )

    def build_adam_clip_blocks(self, loader):
        # Existing SOURCE builds the EK-FAC bases/eigenvalues used for H^{-1}.
        super().build_curvature_blocks(loader)

        for seg_idx, (segment, checkpoints) in enumerate(
            zip(self.segments, self.checkpoints_per_segment)
        ):
            diagonal_curvatures = []
            clip_cache = {}
            for checkpoint in checkpoints:
                diagonal, clip_scale, clip_fraction = (
                    self._diagonal_and_clip(checkpoint, segment, loader)
                )
                diagonal_curvatures.append(diagonal)
                clip_cache[str(checkpoint)] = (clip_scale, clip_fraction)
            segment.diagonal_curvature = {
                name: torch.stack(
                    [value[name] for value in diagonal_curvatures]
                ).mean(dim=0)
                for name in segment.modules_name
            }

            p_checkpoints = self.preconditioner_checkpoints_per_segment[seg_idx]
            lr_weights = self.preconditioner_lr_weights_per_segment[seg_idx]
            lr_sum = float(adam_clip_source_lr_sums_per_segment()[seg_idx])
            if abs(sum(lr_weights) - lr_sum) > max(1e-10, 1e-8 * lr_sum):
                raise ValueError(
                    f"segment {seg_idx} p/c weights sum to {sum(lr_weights)}, "
                    f"expected {lr_sum}"
                )
            weighted_p = None
            weighted_cp = None
            weighted_clip_scale = 0.0
            weighted_clip_fraction = 0.0
            checkpoint_diagnostics = []
            for checkpoint, weight in zip(p_checkpoints, lr_weights):
                preconditioner = self._checkpoint_preconditioner(
                    checkpoint, segment
                )
                cache_key = str(checkpoint)
                if cache_key in clip_cache:
                    clip_scale, clip_fraction = clip_cache[cache_key]
                else:
                    clip_scale, clip_fraction = self._estimate_clip_scale(
                        checkpoint, loader
                    )
                if weighted_p is None:
                    weighted_p = {
                        name: float(weight) * value
                        for name, value in preconditioner.items()
                    }
                    weighted_cp = {
                        name: float(weight) * clip_scale * value
                        for name, value in preconditioner.items()
                    }
                else:
                    for name in segment.modules_name:
                        weighted_p[name].add_(
                            preconditioner[name], alpha=float(weight)
                        )
                        weighted_cp[name].add_(
                            preconditioner[name],
                            alpha=float(weight) * clip_scale,
                        )
                weighted_clip_scale += float(weight) * clip_scale
                weighted_clip_fraction += float(weight) * clip_fraction
                checkpoint_diagnostics.append(
                    {
                        "checkpoint": str(checkpoint),
                        "lr_weight": float(weight),
                        "clip_scale": float(clip_scale),
                        "clip_fraction": float(clip_fraction),
                    }
                )

            segment.adam_preconditioner = {
                name: value / lr_sum for name, value in weighted_p.items()
            }
            segment.effective_adam_clip_preconditioner = {
                name: value / lr_sum for name, value in weighted_cp.items()
            }
            segment.clip_scale = weighted_clip_scale / lr_sum
            segment.clip_fraction = weighted_clip_fraction / lr_sum
            segment.adam_clip_checkpoint_diagnostics = checkpoint_diagnostics
            segment.adam_clip_decay = {}
            with torch.no_grad():
                for name in segment.modules_name:
                    exponent = (
                        lr_sum
                        * segment.effective_adam_clip_preconditioner[name]
                        * segment.diagonal_curvature[name]
                    )
                    segment.adam_clip_decay[name] = torch.exp(
                        -exponent.clamp(min=0.0, max=80.0)
                    )
            self.logger.info(
                "Adam/clipping segment %d/%d: lr-weighted_clip_scale=%.6g "
                "lr-weighted_clipped_fraction=%.4f p_checkpoints=%d lr_sum=%.6g",
                seg_idx + 1,
                len(self.segments),
                segment.clip_scale,
                segment.clip_fraction,
                len(p_checkpoints),
                lr_sum,
            )

        self._restore_final_params()

    @staticmethod
    def _clone_grads(grads):
        return {name: value.clone() for name, value in grads.items()}

    @staticmethod
    def _jacobian_frobenius_rms_normalize(grads, output_dim):
        """Divide each query's Jacobian rows by ||J||_F/sqrt(output_dim)."""
        if not grads:
            raise ValueError("cannot normalize an empty gradient dictionary")
        first = next(iter(grads.values()))
        if first.shape[0] % output_dim != 0:
            raise ValueError(
                f"measurement count {first.shape[0]} is not divisible by {output_dim}"
            )
        row_norm_sq = None
        for value in grads.values():
            contribution = value.float().square().flatten(1).sum(dim=1)
            row_norm_sq = (
                contribution if row_norm_sq is None else row_norm_sq + contribution
            )
        num_queries = first.shape[0] // output_dim
        frobenius_sq = row_norm_sq.reshape(num_queries, output_dim).sum(dim=1)
        denominator = torch.sqrt(frobenius_sq / float(output_dim)).clamp_min(
            ADAM_CLIP_SOURCE_NORM_EPS
        ).repeat_interleave(output_dim)
        return {
            name: value / denominator.reshape(-1, *([1] * (value.ndim - 1)))
            for name, value in grads.items()
        }

    @staticmethod
    def _apply_diagonal(grads, factors):
        return {name: grads[name] * factors[name].unsqueeze(0) for name in grads}

    @staticmethod
    def _inverse_factor(eigenvalues):
        relative = (
            eigenvalues.detach().abs().mean()
            * ADAM_CLIP_SOURCE_EIGENVALUE_RELATIVE_FLOOR
        )
        floor = torch.clamp(
            relative, min=ADAM_CLIP_SOURCE_EIGENVALUE_ABSOLUTE_FLOOR
        )
        return torch.reciprocal(eigenvalues.clamp_min(floor))

    def compute_score_variants_with_loader(self, test_loader, train_loader):
        if not self.segments:
            raise RuntimeError("call build_adam_clip_blocks first")
        self._restore_final_params()
        num_test = len(test_loader.dataset)
        num_train = len(train_loader.dataset)
        score_tables = {
            variant: torch.zeros(
                (num_test, num_train),
                dtype=self.score_dtype,
                device=self.task.device,
            )
            for variant in ADAM_CLIP_SOURCE_METHODS
        }

        num_processed_test = 0
        for test_batch in test_loader:
            test_batch_size = self.task.get_batch_size(test_batch)
            self.logger.info(
                "Processing Adam/clipping test batch [%d, %d) of %d.",
                num_processed_test,
                num_processed_test + test_batch_size,
                num_test,
            )
            measurement_grads = self._measurement_grads_dict(
                test_batch, self.final_func_params, self.final_func_buffers
            )
            example_segment = self.segments[0]
            module_grads = {}
            with torch.no_grad():
                for name, module in zip(
                    example_segment.modules_name, example_segment.modules
                ):
                    module_grads[name] = make_grads_dict_to_matrix(
                        module=module,
                        module_name=name,
                        grads_dict=measurement_grads,
                        remove_grads=True,
                    ).to(dtype=self.grads_dtype)
            del measurement_grads
            running = {
                "unnormalized": self._clone_grads(module_grads),
                "jacobian_fro_rms": self._jacobian_frobenius_rms_normalize(
                    module_grads, ADAM_CLIP_SOURCE_OUTPUT_DIM
                ),
            }
            del module_grads

            for seg_idx in range(len(self.segments) - 1, -1, -1):
                segment = self.segments[seg_idx]
                one_minus_decay = {
                    name: 1.0 - segment.adam_clip_decay[name]
                    for name in segment.modules_name
                }
                preconditioned = {}
                for variant in ADAM_CLIP_SOURCE_METHODS:
                    gated = self._apply_diagonal(
                        running[variant], one_minus_decay
                    )
                    transformed = self._apply_matrix_function(
                        gated, segment, self._inverse_factor
                    )
                    preconditioned[variant] = {
                        name: transformed[name].reshape(test_batch_size, -1)
                        for name in segment.modules_name
                    }
                    del gated, transformed

                self._accumulate_score_variants(
                    segment=segment,
                    seg_idx=seg_idx,
                    preconditioned=preconditioned,
                    train_loader=train_loader,
                    score_tables=score_tables,
                    num_processed_test=num_processed_test,
                    test_batch_size=test_batch_size,
                )
                del preconditioned
                if seg_idx > 0:
                    for variant in ADAM_CLIP_SOURCE_METHODS:
                        running[variant] = self._apply_diagonal(
                            running[variant], segment.adam_clip_decay
                        )
            num_processed_test += test_batch_size

        self._restore_final_params()
        return score_tables

    def _accumulate_score_variants(
        self,
        segment,
        seg_idx,
        preconditioned,
        train_loader,
        score_tables,
        num_processed_test,
        test_batch_size,
    ):
        checkpoints = self.checkpoints_per_segment[seg_idx]
        weight = 1.0 / float(len(checkpoints))
        for checkpoint in checkpoints:
            self._load_checkpoint(checkpoint)
            params = dict(self.model.named_parameters())
            buffers = dict(self.model.named_buffers())
            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)
                grads_dict = self._train_loss_grads_dict(
                    train_batch, params, buffers
                )
                with torch.no_grad():
                    for name, module in zip(
                        segment.modules_name, segment.modules
                    ):
                        train_grads = make_grads_dict_to_matrix(
                            module=module,
                            module_name=name,
                            grads_dict=grads_dict,
                            remove_grads=True,
                        ).reshape(train_batch_size, -1).to(dtype=self.grads_dtype)
                        for variant in ADAM_CLIP_SOURCE_METHODS:
                            score_tables[variant][
                                num_processed_test : num_processed_test
                                + test_batch_size,
                                num_processed_train : num_processed_train
                                + train_batch_size,
                            ].addmm_(
                                preconditioned[variant][name],
                                train_grads.t(),
                                alpha=weight,
                            )
                        del train_grads
                num_processed_train += train_batch_size
                del grads_dict

    def diagnostics(self):
        return [
            {
                "segment": index,
                "clip_scale": float(segment.clip_scale),
                "clip_fraction": float(segment.clip_fraction),
                "decay_min": float(
                    min(value.min().item() for value in segment.adam_clip_decay.values())
                ),
                "decay_max": float(
                    max(value.max().item() for value in segment.adam_clip_decay.values())
                ),
                "preconditioner_checkpoint_diagnostics": (
                    segment.adam_clip_checkpoint_diagnostics
                ),
            }
            for index, segment in enumerate(self.segments)
        ]
