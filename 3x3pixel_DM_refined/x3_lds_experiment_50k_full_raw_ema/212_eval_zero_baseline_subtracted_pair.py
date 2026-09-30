"""Test exact checkpoint deltas after subtracting zero-gradient AdamW drift."""

import argparse
import importlib
import json

import numpy as np
import torch
from torch.func import functional_call, jvp

from attribution_one_query import build_model, model_paths
from checkpoint_transition_diagnostic_config import *
from dataset_loader import ColorGridDataset
from train_worker import lr_at


transition = importlib.import_module("204_run_checkpoint_transition_diagnostic_shard")
endpoint = importlib.import_module("206_run_checkpoint_endpoint_adam_diagnostic")


def tuple_metrics(predicted, actual):
    return transition.parameter_agreement(predicted, actual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pair-index", type=int, default=16)
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--query-batch-size", type=int, default=CTD_QUERY_BATCH_SIZE)
    args = parser.parse_args()
    if not 0 <= args.pair_index < 49:
        raise ValueError("--pair-index must be in [0, 48]")

    transition.configure_training_precision()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    start_model, _, start_checkpoint = build_model(paths[args.pair_index], "raw", device)
    target_model, _, target_checkpoint = build_model(paths[args.pair_index + 1], "raw", device)
    zero_model, zero_optimizer = endpoint.make_shadow(
        paths[args.pair_index], start_checkpoint, device
    )
    start_names = tuple(name for name, _ in start_model.named_parameters())
    start_state = dict(start_model.named_parameters())
    target_state = dict(target_model.named_parameters())

    _, batches_per_epoch = transition.checkpoint_orders(
        dataset, device, [args.pair_index + 1]
    )
    total_steps = EPOCHS * batches_per_epoch
    start_step = int(start_checkpoint["global_step"])
    target_step = int(target_checkpoint["global_step"])
    for global_step in range(start_step, target_step):
        learning_rate = lr_at(global_step, total_steps)
        for group in zero_optimizer.param_groups:
            group["lr"] = learning_rate
        zero_optimizer.zero_grad(set_to_none=True)
        for parameter in zero_model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        zero_optimizer.step()

    zero_state = dict(zero_model.named_parameters())
    exact_delta = tuple(
        target_state[name].detach() - start_state[name].detach() for name in start_names
    )
    baseline_delta = tuple(
        zero_state[name].detach() - start_state[name].detach() for name in start_names
    )
    data_delta = tuple(
        exact - baseline for exact, baseline in zip(exact_delta, baseline_delta)
    )

    query_ids = parse_integer_selection(args.query_ids, CTD_QUERY_IDS)
    with open(QUERY_DIR / "manifest.json") as handle:
        records = [
            record for record in json.load(handle)
            if int(record["query_id"]) in query_ids and record["family"] == args.family
        ]
    bank = transition.query_bank(records, dataset, device)
    transition.configure_evaluation_precision()
    start_model = start_model.double().eval()
    names = tuple(name for name, _ in start_model.named_parameters())
    initial = tuple(parameter.detach() for parameter in start_model.parameters())
    target = tuple(target_state[name].to(device=device, dtype=torch.float64) for name in names)
    tangents = {
        "exact_parameter_delta": tuple(value.double() for value in exact_delta),
        "zero_gradient_baseline": tuple(value.double() for value in baseline_delta),
        "baseline_subtracted_data_delta": tuple(value.double() for value in data_delta),
    }
    arrays = {method: [] for method in tangents}
    actual_norm_chunks = []
    for start in range(0, len(bank["states"]), args.query_batch_size):
        end = min(start + args.query_batch_size, len(bank["states"]))
        x = torch.from_numpy(bank["states"][start:end]).to(device=device, dtype=torch.float64)
        timestep = torch.from_numpy(bank["timesteps"][start:end]).to(device=device, dtype=torch.long)
        condition = torch.from_numpy(bank["conditions"][start:end]).to(device=device, dtype=torch.float64)

        def prediction(*parameters):
            return functional_call(start_model, dict(zip(names, parameters)), (x, timestep, condition))

        with torch.no_grad():
            actual = prediction(*target) - prediction(*initial)
        actual_norm_chunks.append(
            actual.detach().double().flatten(1).norm(dim=1).cpu().numpy()
        )
        for method, tangent in tangents.items():
            predicted = jvp(prediction, initial, tangent)[1]
            arrays[method].append(transition.point_metrics(predicted, actual))

    output = {
        "pair_index": args.pair_index,
        "start_epoch": int(start_checkpoint["epoch"]),
        "target_epoch": int(target_checkpoint["epoch"]),
        "optimizer_steps": target_step - start_step,
        "parameter": {
            "baseline_vs_exact": tuple_metrics(baseline_delta, exact_delta),
            "data_vs_exact": tuple_metrics(data_delta, exact_delta),
        },
        "response": {},
    }
    print("zero-gradient baseline subtraction", flush=True)
    print("method                              cosine    vector-error  mag-error  ratio")
    actual_norm = np.concatenate(actual_norm_chunks)
    for method, chunks in arrays.items():
        keys = chunks[0].keys()
        merged = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
        result = {
            "vector_cosine_mean": float(np.nanmean(merged["vector_cosine"])),
            "vector_relative_error_mean": float(np.nanmean(merged["vector_relative_error"])),
            "magnitude_relative_error_mean": float(np.nanmean(merged["magnitude_relative_error"])),
            "magnitude_ratio": float(merged["predicted_l2"].sum() / np.maximum(actual_norm.sum(), 1e-30)),
        }
        output["response"][method] = result
        print(f"{method:35s} {result['vector_cosine_mean']:+.6f}  {result['vector_relative_error_mean']:.6f}      {result['magnitude_relative_error_mean']:.6f}  {result['magnitude_ratio']:.6f}")
    root = ROOT / "zero_gradient_baseline_subtracted_pair_10q_100t"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{args.family}_pair_{args.pair_index:02d}.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"parameter baseline/exact norm ratio={output['parameter']['baseline_vs_exact']['predicted_norm'] / output['parameter']['baseline_vs_exact']['actual_norm']:.6f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
