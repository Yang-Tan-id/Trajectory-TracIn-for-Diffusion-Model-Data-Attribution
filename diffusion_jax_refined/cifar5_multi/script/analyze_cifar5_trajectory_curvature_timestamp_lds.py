#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_cifar5_multi_experiment import (
    Job,
    gpu_env,
    parse_gpus,
    run_parallel_jobs,
    slot_for,
    worker_gpus,
)
from run_cifar5_multi_random_prompted_queries import build_query_specs, query_tag
from run_cifar5_multi_traj_temporal_thirds import SEGMENTS, segment_position_map
from run_cifar5_multi_traj_tracin_norm_sweep import (
    Component,
    parse_ranges,
    query_artifact,
    train_shard,
)


TARGET_ALIASES = {
    "endpoint_counterfactual": "endpoint_counterfactual",
    "endpoint_contarfactual": "endpoint_counterfactual",
    "noise_trajectory": "noise_trajectory",
    "traj_counterfactual": "traj_counterfactual",
    "traj_contarfactual": "traj_counterfactual",
    "simple_loss": "simple_loss",
}
TARGETS = (
    "endpoint_counterfactual",
    "noise_trajectory",
    "traj_counterfactual",
    "simple_loss",
)
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(prediction) & np.isfinite(target)
    if int(mask.sum()) < 2:
        return float("nan")
    x = rankdata(prediction[mask])
    y = rankdata(target[mask])
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def output_root(root: Path, args: argparse.Namespace) -> Path:
    return root / "result" / args.experiment / "diagnostics" / args.output_name


def score_shard_path(root: Path, args: argparse.Namespace, start: int, end: int) -> Path:
    return output_root(root, args) / "score_shards" / args.variant / f"range_{start}_{end}.npz"


def components(args: argparse.Namespace) -> tuple[Component, ...]:
    return (
        Component("base", args.base_train_namespace, args.base_query_namespace, args.output_name),
        Component("addon_a", args.addon_a_train_namespace, args.addon_a_query_namespace, args.output_name),
        Component("addon_b", args.addon_b_train_namespace, args.addon_b_query_namespace, args.output_name),
    )


def load_query(path: Path) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["query_features"], dtype=np.float32)
        ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
    mapping = {
        (int(ckpt), int(timestep)): i
        for i, (ckpt, timestep) in enumerate(zip(ckpts, timesteps))
    }
    return features, mapping


def score_worker(root: Path, args: argparse.Namespace, start: int, end: int) -> None:
    output = score_shard_path(root, args, start, end)
    specs = build_query_specs(args)
    positions = sorted(segment_position_map())
    position_index = {position: i for i, position in enumerate(positions)}
    result = None
    score_indices = None
    term_counts = np.zeros(len(positions), dtype=np.int32)

    for component in components(args):
        print(f"[timestamp-lds] loading queries: {component.query_namespace}", flush=True)
        query_payloads = [
            load_query(query_artifact(root, args, spec, component.query_namespace))
            for spec in specs
        ]
        query_features = [item[0] for item in query_payloads]
        query_maps = [item[1] for item in query_payloads]

        path = train_shard(root, args, component, start, end)
        print(f"[timestamp-lds] loading train once: {path}", flush=True)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            snapshot_positions = np.asarray(payload["snapshot_positions"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)

        if result is None:
            result = np.zeros((len(specs), len(positions), len(indices)), dtype=np.float64)
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")

        for term_i, (ckpt, timestep, position) in enumerate(
            zip(ckpts, timesteps, snapshot_positions)
        ):
            timestamp_i = position_index.get(int(position))
            if timestamp_i is None:
                continue
            query_rows = []
            for features, mapping in zip(query_features, query_maps):
                query_i = mapping.get((int(ckpt), int(timestep)))
                if query_i is None:
                    raise ValueError(
                        f"missing query term component={component.label} "
                        f"ckpt={int(ckpt)} timestep={int(timestep)} position={int(position)}"
                    )
                query_rows.append(features[query_i])
            query_matrix = np.stack(query_rows).astype(np.float32, copy=False)
            train_term = np.asarray(train[term_i], dtype=np.float32)

            if args.variant in ("query_l2", "query_train_l2"):
                norms = np.linalg.norm(query_matrix, axis=1)
                query_matrix = query_matrix / np.maximum(norms, args.normalize_eps)[:, None]
            dot = train_term @ query_matrix.T
            if args.variant in ("train_l2", "query_train_l2"):
                train_norms = np.linalg.norm(train_term, axis=1)
                dot = dot / np.maximum(train_norms, args.normalize_eps)[:, None]
            result[:, timestamp_i, :] += (float(weights[term_i]) * dot).T
            term_counts[timestamp_i] += 1
            if (term_i + 1) % 100 == 0 or term_i + 1 == len(train):
                print(
                    f"[timestamp-lds] {component.label} term {term_i + 1}/{len(train)}",
                    flush=True,
                )

        del train, query_features, query_maps, query_payloads
        gc.collect()

    if result is None or score_indices is None:
        raise RuntimeError("no train artifacts were loaded")
    if not np.all(term_counts == 49):
        bad = [(positions[i], int(count)) for i, count in enumerate(term_counts) if count != 49]
        raise ValueError(f"expected 49 checkpoint terms per timestamp; bad={bad}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        scores=result,
        score_indices=score_indices,
        positions=np.asarray(positions, dtype=np.int32),
        timesteps=np.asarray([999 - position for position in positions], dtype=np.int32),
        seeds=np.asarray([int(spec["initial_seed"]) for spec in specs], dtype=np.int32),
        queries=np.asarray([query_tag(str(spec["query"])) for spec in specs]),
        term_counts=term_counts,
        variant=np.asarray(args.variant),
    )
    print(f"[timestamp-lds] wrote {output}", flush=True)


def load_full_scores(
    root: Path, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranges = parse_ranges(args.score_index_ranges, size=args.size)
    score_blocks = []
    index_blocks = []
    positions = None
    timesteps = None
    for start, end in ranges:
        path = score_shard_path(root, args, start, end)
        with np.load(path, allow_pickle=False) as payload:
            score_blocks.append(np.asarray(payload["scores"], dtype=np.float64))
            index_blocks.append(np.asarray(payload["score_indices"], dtype=np.int64))
            current_positions = np.asarray(payload["positions"], dtype=np.int32)
            current_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        if positions is None:
            positions = current_positions
            timesteps = current_timesteps
        elif not np.array_equal(positions, current_positions):
            raise ValueError(f"positions differ in {path}")
    indices = np.concatenate(index_blocks)
    scores = np.concatenate(score_blocks, axis=2)
    order = np.argsort(indices)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("duplicate score indices across shards")
    assert positions is not None and timesteps is not None
    return scores[:, :, order], indices[order], positions, timesteps


def cached_targets(
    root: Path, args: argparse.Namespace, spec: dict[str, int | str]
) -> dict[str, Path]:
    base = (
        root / "result" / args.experiment / "eval" / "prompted_solo"
        / f"query_{query_tag(str(spec['query']))}"
        / f"initial_seed_{int(spec['initial_seed'])}"
        / args.base_train_namespace / "lds" / "traj_tracin"
    )
    found = {}
    for path in base.glob("*/lds_results.csv"):
        canonical = TARGET_ALIASES.get(path.parent.name)
        if canonical:
            found[canonical] = path
    return found


def subset_predictions(
    score_by_timestamp: np.ndarray,
    score_indices: np.ndarray,
    target_rows: list[dict[str, str]],
) -> np.ndarray:
    index_to_column = {int(index): i for i, index in enumerate(score_indices)}
    indicators = np.zeros((len(target_rows), len(score_indices)), dtype=np.float32)
    for row_i, row in enumerate(target_rows):
        kept = np.asarray(
            np.load(Path(row["subset_dir"]) / "kept_attribution_indices.npy"),
            dtype=np.int64,
        ).reshape(-1)
        columns = [index_to_column[int(index)] for index in kept if int(index) in index_to_column]
        indicators[row_i, columns] = 1.0
    return -1.0 * (indicators @ score_by_timestamp.T)


def calculate_timestamp_lds(
    root: Path,
    args: argparse.Namespace,
    scores: np.ndarray,
    score_indices: np.ndarray,
    positions: np.ndarray,
    timesteps: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    specs = build_query_specs(args)
    position_segments = segment_position_map()
    for spec_i, spec in enumerate(specs):
        targets = cached_targets(root, args, spec)
        if set(targets) != set(TARGETS):
            raise FileNotFoundError(
                f"expected four cached LDS targets for query={spec['query']} "
                f"seed={spec['initial_seed']}; found={sorted(targets)}"
            )
        for target in TARGETS:
            target_rows = read_csv(targets[target])
            true = np.asarray([float(row["true_f"]) for row in target_rows], dtype=np.float64)
            predictions = subset_predictions(scores[spec_i], score_indices, target_rows)
            for timestamp_i, (position, timestep) in enumerate(zip(positions, timesteps)):
                value = spearman(predictions[:, timestamp_i], true)
                rows.append(
                    {
                        "seed": int(spec["initial_seed"]),
                        "query": f"query_{query_tag(str(spec['query']))}",
                        "target": target,
                        "position": int(position),
                        "timestep": int(timestep),
                        "segment": position_segments[int(position)],
                        "variant": args.variant,
                        "lds": value,
                        "num_models": len(target_rows),
                    }
                )
    return rows


def trajectory_path(root: Path, args: argparse.Namespace, spec: dict[str, int | str]) -> Path:
    return (
        root / "result" / args.experiment / "sample" / "cifar"
        / f"prompt_{query_tag(str(spec['query']))}"
        / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
        / f"seed_{int(spec['initial_seed']):06d}" / "trajectory_xt.npy"
    )


def curvature_metrics(path: Path) -> dict[str, float | int]:
    trajectory = np.asarray(np.load(path), dtype=np.float32)
    states = trajectory.reshape(trajectory.shape[0], -1).astype(np.float64)
    steps = np.diff(states, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    valid = step_norms > 1e-12
    adjacent_valid = valid[:-1] & valid[1:]
    cosines = np.sum(steps[:-1] * steps[1:], axis=1) / np.maximum(
        step_norms[:-1] * step_norms[1:], 1e-12
    )
    cosines = np.clip(cosines[adjacent_valid], -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    chord_vector = states[-1] - states[0]
    chord = float(np.linalg.norm(chord_vector))
    path_length = float(step_norms.sum())
    if chord > 1e-12:
        unit = chord_vector / chord
        offsets = states - states[0]
        projections = (offsets @ unit)[:, None] * unit[None, :]
        deviations = np.linalg.norm(offsets - projections, axis=1)
    else:
        deviations = np.full(len(states), np.nan)
    return {
        "trajectory_states": int(len(states)),
        "path_length": path_length,
        "endpoint_distance": chord,
        "tortuosity": path_length / max(chord, 1e-12),
        "mean_turn_angle_deg": float(np.mean(angles)) if len(angles) else float("nan"),
        "median_turn_angle_deg": float(np.median(angles)) if len(angles) else float("nan"),
        "mean_turn_cosine": float(np.mean(cosines)) if len(cosines) else float("nan"),
        "mean_line_deviation_over_chord": float(np.nanmean(deviations)) / max(chord, 1e-12),
        "max_line_deviation_over_chord": float(np.nanmax(deviations)) / max(chord, 1e-12),
    }


def ranking_scores(root: Path, args: argparse.Namespace) -> dict[tuple[int, str], float]:
    by_query = defaultdict(list)
    algorithm = {
        "raw": "traj_tracin",
        "query_l2": "traj_tracin_query_normalized",
        "train_l2": "traj_tracin_train_l2_normalized",
        "query_train_l2": "traj_tracin_query_train_l2_normalized",
    }[args.variant]
    for spec in build_query_specs(args):
        base = (
            root / "result" / args.experiment / "eval" / "prompted_solo"
            / f"query_{query_tag(str(spec['query']))}"
            / f"initial_seed_{int(spec['initial_seed'])}"
            / args.ranking_namespace / "lds" / algorithm
        )
        for path in base.glob("*/lds_summary.json"):
            by_query[(int(spec["initial_seed"]), query_tag(str(spec["query"])))].append(
                float(json.loads(path.read_text())["lds_spearman"])
            )
    return {key: float(np.mean(values)) for key, values in by_query.items() if values}


def analyze_curvature(root: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    rankings = ranking_scores(root, args)
    rows = []
    for spec in build_query_specs(args):
        path = trajectory_path(root, args, spec)
        if not path.is_file():
            print(f"[trajectory missing] {path}", flush=True)
            continue
        seed = int(spec["initial_seed"])
        tag = query_tag(str(spec["query"]))
        row: dict[str, object] = {
            "seed": seed,
            "query": f"query_{tag}",
            "ranking_lds": rankings.get((seed, tag), float("nan")),
            "trajectory_path": str(path),
        }
        row.update(curvature_metrics(path))
        rows.append(row)
    return rows


def print_report(
    args: argparse.Namespace,
    curvature_rows: list[dict[str, object]],
    lds_rows: list[dict[str, object]],
) -> None:
    focus = [row for row in curvature_rows if int(row["seed"]) == args.focus_seed]
    candidates = [
        row for row in curvature_rows
        if int(row["seed"]) != args.focus_seed and math.isfinite(float(row["ranking_lds"]))
    ]
    comparisons = sorted(candidates, key=lambda row: float(row["ranking_lds"]), reverse=True)[
        : args.top_comparisons
    ]
    selected = focus + comparisons
    print("\nTrajectory curvature: larger angle/tortuosity/deviation means more curved")
    print(
        f"{'seed':>6s} {'query':32s} {'LDS':>8s} {'turn_deg':>10s} "
        f"{'tortuosity':>11s} {'mean_dev':>10s} {'max_dev':>10s}"
    )
    print("-" * 98)
    for row in selected:
        print(
            f"{int(row['seed']):6d} {str(row['query']):32s} "
            f"{float(row['ranking_lds']):8.3f} {float(row['mean_turn_angle_deg']):10.3f} "
            f"{float(row['tortuosity']):11.3f} "
            f"{float(row['mean_line_deviation_over_chord']):10.3f} "
            f"{float(row['max_line_deviation_over_chord']):10.3f}"
        )

    good_seeds = {int(row["seed"]) for row in comparisons}
    print("\nPer-timestamp LDS: seed 1010 versus mean of the selected high-LDS queries")
    print(
        f"{'segment':14s} {'pos':>4s} {'t':>4s} {'1010-end':>9s} {'1010-noise':>11s} "
        f"{'1010-traj':>10s} {'1010-simple':>12s} {'1010-all':>9s} {'good-all':>9s}"
    )
    print("-" * 108)
    positions = sorted({int(row["position"]) for row in lds_rows})
    for position in positions:
        current = [row for row in lds_rows if int(row["position"]) == position]
        focus_rows = [row for row in current if int(row["seed"]) == args.focus_seed]
        focus_by_target = {
            target: [float(row["lds"]) for row in focus_rows if row["target"] == target]
            for target in TARGETS
        }
        values = [value for target_values in focus_by_target.values() for value in target_values]
        good = [float(row["lds"]) for row in current if int(row["seed"]) in good_seeds]
        first = current[0]
        means = [
            float(np.mean(focus_by_target[target])) if focus_by_target[target] else float("nan")
            for target in TARGETS
        ]
        print(
            f"{str(first['segment']):14s} {position:4d} {int(first['timestep']):4d} "
            f"{means[0]:9.3f} {means[1]:11.3f} {means[2]:10.3f} {means[3]:12.3f} "
            f"{float(np.mean(values)):9.3f} {float(np.mean(good)) if good else float('nan'):9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CIFAR5 trajectory curvature and calculate LDS for each of 30 timestamps."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="cifar5_multi_exp1")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--random-query-seed", type=int, default=0)
    parser.add_argument("--initial-seed-start", type=int, default=1000)
    parser.add_argument("--initial-seeds", default="")
    parser.add_argument("--extra-prompted-queries", default="")
    parser.add_argument("--extra-initial-seed", type=int, default=0)
    parser.add_argument("--focus-seed", type=int, default=1010)
    parser.add_argument("--top-comparisons", type=int, default=3)
    parser.add_argument("--variant", choices=VARIANTS, default="train_l2")
    parser.add_argument(
        "--ranking-namespace",
        default="raw_nextckpt_school_traj_combined30_mid10_ab",
    )
    parser.add_argument(
        "--output-name",
        default="trajectory_curvature_timestamp_lds_combined30",
    )
    parser.add_argument("--base-train-namespace", default="raw_nextckpt_school_traj_aligned_10x10")
    parser.add_argument("--base-query-namespace", default="raw_nextckpt_school_traj_10x10")
    parser.add_argument("--addon-a-train-namespace", default="raw_nextckpt_school_traj_addon_mid10")
    parser.add_argument("--addon-a-query-namespace", default="raw_nextckpt_school_traj_addon_mid10_exact10")
    parser.add_argument("--addon-b-train-namespace", default="raw_nextckpt_school_traj_addon_mid10_b")
    parser.add_argument("--addon-b-query-namespace", default="raw_nextckpt_school_traj_addon_mid10_b_exact10")
    parser.add_argument("--score-index-ranges", default="1-2500,2501-5000,5001-7500,7501-10000")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default="local")
    parser.add_argument("--use-task-affinity", action="store_true")
    parser.add_argument("--normalize-eps", type=float, default=1e-8)
    parser.add_argument("--worker-range", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent.parent
    if args.worker_range:
        start, end = parse_ranges(args.worker_range, size=args.size)[0]
        score_worker(root, args, start, end)
        return

    ranges = parse_ranges(args.score_index_ranges, size=args.size)
    missing_ranges = [
        (start, end)
        for start, end in ranges
        if not score_shard_path(root, args, start, end).is_file()
    ]
    if missing_ranges and not args.execute:
        print(f"[dry-run] missing timestamp score shards: {missing_ranges}")
        print("[dry-run] add --execute to calculate them")
        return
    if missing_ranges:
        gpus = parse_gpus(args)
        worker_gpu_ids = worker_gpus(args, gpus)
        jobs = []
        for range_i, (start, end) in enumerate(missing_ranges):
            slot = slot_for(range_i, len(worker_gpu_ids))
            jobs.append(
                Job(
                    name=f"timestamp_lds_{start}_{end}",
                    cmd=[
                        sys.executable,
                        str(Path(__file__).resolve()),
                        *sys.argv[1:],
                        "--worker-range",
                        f"{start}-{end}",
                    ],
                    cwd=repo_root,
                    env=gpu_env(os.environ.copy(), worker_gpu_ids[slot]),
                    log_path=(
                        output_root(root, args) / "logs"
                        / f"range_{start}_{end}_gpu_{worker_gpu_ids[slot]}.log"
                    ),
                    slot=slot,
                )
            )
        run_parallel_jobs(
            jobs,
            args=args,
            execute=True,
            max_parallel=max(1, min(args.max_parallel, len(worker_gpu_ids))),
        )

    scores, score_indices, positions, timesteps = load_full_scores(root, args)
    lds_rows = calculate_timestamp_lds(
        root, args, scores, score_indices, positions, timesteps
    )
    curvature_rows = analyze_curvature(root, args)
    out = output_root(root, args)
    write_csv(out / "timestamp_lds.csv", lds_rows)
    write_csv(out / "trajectory_curvature.csv", curvature_rows)
    print_report(args, curvature_rows, lds_rows)
    print(f"\n[saved] {out / 'timestamp_lds.csv'}")
    print(f"[saved] {out / 'trajectory_curvature.csv'}")


if __name__ == "__main__":
    main()
