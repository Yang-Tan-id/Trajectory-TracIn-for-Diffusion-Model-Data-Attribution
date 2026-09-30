"""Predict checkpoint transitions using only fixed endpoint gradients.

Unlike the full replay control, the gradient models never move. Three shadow
AdamW optimizers receive gradients evaluated at the start checkpoint, target
checkpoint, or their endpoint-trapezoid average.
"""

import argparse
import importlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from attribution_one_query import build_model, model_paths, preload_dataset
from checkpoint_endpoint_adam_diagnostic_config import *
from dataset_loader import ColorGridDataset
from forward_loss_alignment_config import replay_noise_path, replay_t_path
from train_worker import lr_at
from x3_endpoint_das_jax_logic_pytorch import make_torch_generator


transition = importlib.import_module("204_run_checkpoint_transition_diagnostic_shard")


def make_shadow(path, checkpoint, device):
    model, _, _ = build_model(path, "raw", device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=(ADAM_B1, ADAM_B2),
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model, optimizer


def copy_gradient(target_model, source_gradients):
    for parameter, gradient in zip(target_model.parameters(), source_gradients):
        parameter.grad = gradient.detach().clone()


def frozen_gradient(model, xt, timestep, condition, noise):
    model.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(xt, timestep, condition), noise)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    return tuple(parameter.grad.detach().clone() for parameter in model.parameters()), float(norm)


def evaluate(
    start_model,
    target_state,
    approximation_states,
    bank,
    device,
    batch_size,
):
    transition.configure_evaluation_precision()
    start_model = start_model.double().eval()
    names = tuple(name for name, _ in start_model.named_parameters())
    initial = tuple(parameter.detach() for parameter in start_model.parameters())
    target = tuple(
        target_state[name].to(device=device, dtype=torch.float64) for name in names
    )
    exact_delta = tuple(after - before for before, after in zip(initial, target))
    tangents = {"exact_parameter_delta_start_jvp": exact_delta}
    for method, state in approximation_states.items():
        endpoint = tuple(
            state[name].to(device=device, dtype=torch.float64) for name in names
        )
        tangents[method] = tuple(
            after - before for before, after in zip(initial, endpoint)
        )

    total = len(bank["states"])
    arrays = {"actual_l2": np.empty(total, dtype=np.float64)}
    for method in CEAD_METHODS:
        for metric in (
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        ):
            arrays[f"{method}_{metric}"] = np.empty(total, dtype=np.float64)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        x = torch.from_numpy(bank["states"][start:end]).to(
            device=device, dtype=torch.float64
        )
        timestep = torch.from_numpy(bank["timesteps"][start:end]).to(
            device=device, dtype=torch.long
        )
        condition = torch.from_numpy(bank["conditions"][start:end]).to(
            device=device, dtype=torch.float64
        )

        def prediction(*parameter_values):
            return functional_call(
                start_model,
                dict(zip(names, parameter_values)),
                (x, timestep, condition),
            )

        with torch.no_grad():
            actual = prediction(*target) - prediction(*initial)
        arrays["actual_l2"][start:end] = (
            actual.flatten(1).norm(dim=1).cpu().numpy()
        )
        for method, tangent in tangents.items():
            predicted = jvp(prediction, initial, tangent)[1]
            for metric, value in transition.point_metrics(predicted, actual).items():
                arrays[f"{method}_{metric}"][start:end] = value
    return arrays, exact_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--query-batch-size", type=int, default=CTD_QUERY_BATCH_SIZE)
    parser.add_argument("--loss-mc", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.pair_index < 49:
        raise ValueError("--pair-index must be in [0, 48]")
    if args.loss_mc <= 0:
        raise ValueError("--loss-mc must be positive")

    transition.configure_training_precision()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x_all, condition_all = preload_dataset(dataset, args.family, device)
    schedule = base.make_linear_schedule(T, device=device)
    paths = model_paths(args.family)
    start_model, _, start_checkpoint = build_model(
        paths[args.pair_index], "raw", device
    )
    target_index = args.pair_index + 1
    target_model, _, target_checkpoint = build_model(paths[target_index], "raw", device)
    start_model.train()
    target_model.train()

    shadow_start, optimizer_start = make_shadow(
        paths[args.pair_index], start_checkpoint, device
    )
    shadow_target, optimizer_target = make_shadow(
        paths[args.pair_index], start_checkpoint, device
    )
    shadow_trapezoid, optimizer_trapezoid = make_shadow(
        paths[args.pair_index], start_checkpoint, device
    )
    shadows = (
        (shadow_start, optimizer_start),
        (shadow_target, optimizer_target),
        (shadow_trapezoid, optimizer_trapezoid),
    )

    query_ids = parse_integer_selection(args.query_ids, CTD_QUERY_IDS)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [
        record
        for record in manifest
        if int(record["query_id"]) in query_ids and record["family"] == args.family
    ]
    if not records:
        raise ValueError(f"no selected queries for family={args.family}")
    bank = transition.query_bank(records, dataset, device)
    orders, batches_per_epoch = transition.checkpoint_orders(
        dataset, device, [target_index]
    )
    replay_t = np.load(replay_t_path(), mmap_mode="r")
    replay_noise = np.load(replay_noise_path(), mmap_mode="r")
    total_steps = EPOCHS * batches_per_epoch
    global_step = int(start_checkpoint["global_step"])
    clip_norms = {"start": [], "target": []}
    started = time.perf_counter()

    for event_index, epoch_batches in enumerate(orders[target_index]):
        for batch_position, indices_np in enumerate(epoch_batches):
            indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
            x = x_all.index_select(0, indices)
            condition = condition_all.index_select(0, indices)
            if args.loss_mc == 1:
                timestep = torch.from_numpy(
                    np.array(
                        replay_t[target_index, indices_np, event_index], copy=True
                    )
                ).to(device=device, dtype=torch.long)
                noise = torch.from_numpy(
                    np.array(
                        replay_noise[target_index, indices_np, event_index], copy=True
                    )
                ).to(device=device, dtype=x.dtype)
                x_loss = x
                condition_loss = condition
            else:
                count = len(indices_np)
                generator = make_torch_generator(
                    device,
                    TRAIN_SEED,
                    "checkpoint_endpoint_adam_loss_mc",
                    args.family,
                    args.pair_index,
                    event_index,
                    batch_position,
                    args.loss_mc,
                )
                timestep = torch.randint(
                    0,
                    T,
                    (count, args.loss_mc),
                    generator=generator,
                    device=device,
                    dtype=torch.long,
                ).reshape(-1)
                noise = torch.randn(
                    (count, args.loss_mc, *x.shape[1:]),
                    generator=generator,
                    device=device,
                    dtype=x.dtype,
                ).reshape(-1, *x.shape[1:])
                x_loss = (
                    x[:, None]
                    .expand(count, args.loss_mc, *x.shape[1:])
                    .reshape(-1, *x.shape[1:])
                )
                condition_loss = (
                    condition[:, None]
                    .expand(count, args.loss_mc, condition.shape[-1])
                    .reshape(-1, condition.shape[-1])
                )
            xt = base.q_sample(x_loss, timestep, noise, schedule)

            start_gradient, start_norm = frozen_gradient(
                start_model, xt, timestep, condition_loss, noise
            )
            target_gradient, target_norm = frozen_gradient(
                target_model, xt, timestep, condition_loss, noise
            )
            trapezoid_gradient = tuple(
                0.5 * (left + right)
                for left, right in zip(start_gradient, target_gradient)
            )
            clip_norms["start"].append(start_norm)
            clip_norms["target"].append(target_norm)

            learning_rate = lr_at(global_step, total_steps)
            for model, optimizer, gradient in (
                (shadow_start, optimizer_start, start_gradient),
                (shadow_target, optimizer_target, target_gradient),
                (shadow_trapezoid, optimizer_trapezoid, trapezoid_gradient),
            ):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                copy_gradient(model, gradient)
                optimizer.step()
            global_step += 1
        print(
            f"[endpoint-adam gpu={args.gpu}] pair={args.pair_index:02d} "
            f"event={event_index + 1}/{CTD_EVENTS_PER_INTERVAL}",
            flush=True,
        )

    target_state = {
        name: parameter.detach().clone()
        for name, parameter in target_model.named_parameters()
    }
    approximation_states = {
        "frozen_start_gradient_adamw_jvp": {
            name: parameter.detach().clone()
            for name, parameter in shadow_start.named_parameters()
        },
        "frozen_target_gradient_adamw_jvp": {
            name: parameter.detach().clone()
            for name, parameter in shadow_target.named_parameters()
        },
        "frozen_endpoint_trapezoid_adamw_jvp": {
            name: parameter.detach().clone()
            for name, parameter in shadow_trapezoid.named_parameters()
        },
    }
    arrays, exact_delta = evaluate(
        start_model,
        target_state,
        approximation_states,
        bank,
        device,
        args.query_batch_size,
    )
    start_state = {
        name: parameter.detach().to(dtype=torch.float64)
        for name, parameter in start_model.named_parameters()
    }
    parameter_agreement = {}
    exact_by_name = {
        name: value for name, value in zip(start_state, exact_delta)
    }
    for method, state in approximation_states.items():
        predicted = tuple(
            state[name].to(device=device, dtype=torch.float64) - start_state[name]
            for name in start_state
        )
        actual = tuple(exact_by_name[name] for name in start_state)
        parameter_agreement[method] = transition.parameter_agreement(
            predicted, actual
        )

    output_root = cead_root(args.loss_mc)
    output_dir = output_root / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pair_{args.pair_index:02d}.npz"
    metadata_path = output_dir / f"pair_{args.pair_index:02d}.json"
    np.savez_compressed(
        output_path,
        query_ids=bank["query_ids"],
        timestamp_positions=bank["timestamp_positions"],
        **arrays,
    )
    with open(metadata_path, "w") as handle:
        json.dump(
            {
                "family": args.family,
                "pair_index": args.pair_index,
                "start_epoch": int(start_checkpoint["epoch"]),
                "target_epoch": int(target_checkpoint["epoch"]),
                "gradient_models": "fixed endpoint checkpoints; no moving-model replay",
                "loss_mc": args.loss_mc,
                "loss_sampling": (
                    "exact_replayed_training_event"
                    if args.loss_mc == 1
                    else "independent_t_noise_shared_by_start_and_target"
                ),
                "shadow_optimizer": "restored start AdamW state and exact LR schedule",
                "parameter_agreement": parameter_agreement,
                "start_gradient_norm_mean": float(np.mean(clip_norms["start"])),
                "target_gradient_norm_mean": float(np.mean(clip_norms["target"])),
                "elapsed_minutes": (time.perf_counter() - started) / 60.0,
            },
            handle,
            indent=2,
        )
    print(f"[saved] {output_path}", flush=True)
    for method, values in parameter_agreement.items():
        print(
            f"{method:42s} parameter-cos={values['cosine']:+.6f} "
            f"parameter-relerr={values['relative_error']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
