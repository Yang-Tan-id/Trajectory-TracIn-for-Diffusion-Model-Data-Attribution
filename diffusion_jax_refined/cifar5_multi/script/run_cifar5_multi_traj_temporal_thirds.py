#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_cifar5_multi_experiment import Job, gpu_env, parse_gpus, run_parallel_jobs, slot_for, worker_gpus
from run_cifar5_multi_random_prompted_queries import build_query_specs, query_tag
from run_cifar5_multi_traj_tracin_norm_sweep import (
    Component,
    parse_ranges,
    query_artifact,
    score_root,
    train_shard,
    write_scores,
)


VARIANTS = {
    "raw": "",
    "query_l2": "_query_normalized",
    "train_l2": "_train_l2_normalized",
    "query_train_l2": "_query_train_l2_normalized",
}
SEGMENTS = ("initial_noisy", "middle", "end_clean")
BASE_POSITIONS = tuple(np.linspace(0, 999, 10, dtype=np.int32).tolist())
ADDON_A_POSITIONS = tuple(int(i * 999 // 99) for i in (5, 16, 27, 38, 49, 60, 71, 82, 93, 98))
ADDON_B_POSITIONS = tuple(int(i * 999 // 99) for i in (2, 14, 24, 36, 47, 58, 69, 80, 91, 96))


def segment_position_map() -> dict[int, str]:
    positions = sorted(set(BASE_POSITIONS + ADDON_A_POSITIONS + ADDON_B_POSITIONS))
    if len(positions) != 30:
        raise ValueError(f"expected 30 distinct temporal positions, found {len(positions)}")
    return {position: SEGMENTS[i // 10] for i, position in enumerate(positions)}


def score_component(segment: str, variant: str) -> str:
    return f"score_{segment}{VARIANTS[variant]}"


def score_shard_dir(
    root: Path,
    args: argparse.Namespace,
    spec: dict[str, int | str],
    segment: str,
    variant: str,
    start: int,
    end: int,
) -> Path:
    return (
        score_root(root, args, spec, args.output_namespace)
        / score_component(segment, variant)
        / "datapoint_shards"
        / f"range_{start}_{end}"
    )


def score_final_dir(
    root: Path,
    args: argparse.Namespace,
    spec: dict[str, int | str],
    segment: str,
    variant: str,
) -> Path:
    return score_root(root, args, spec, args.output_namespace) / score_component(segment, variant)


def load_query_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = ("query_features", "ckpt_indices", "timesteps")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{path} missing query fields: {missing}")
        return {key: np.asarray(payload[key]) for key in required}


def run_worker(root: Path, args: argparse.Namespace, start: int, end: int) -> None:
    specs = build_query_specs(args)
    position_to_segment = segment_position_map()
    components = (
        Component("base", args.base_train_namespace, args.base_query_namespace, args.output_namespace),
        Component("addon_a", args.addon_a_train_namespace, args.addon_a_query_namespace, args.output_namespace),
        Component("addon_b", args.addon_b_train_namespace, args.addon_b_query_namespace, args.output_namespace),
    )
    num_queries = len(specs)
    scores: dict[tuple[str, str], np.ndarray] = {}
    score_indices = None
    segment_term_counts: dict[tuple[int, str], int] = {}
    skipped_terms: set[tuple[str, int, int]] = set()
    query_eps = float(args.query_normalize_eps)
    train_eps = float(args.train_normalize_eps)

    for component in components:
        print(
            f"[temporal-thirds] loading query namespace once: {component.query_namespace}",
            flush=True,
        )
        query_payloads = [
            load_query_payload(query_artifact(root, args, spec, component.query_namespace))
            for spec in specs
        ]
        query_maps = [
            {
                (int(ckpt), int(timestep)): i
                for i, (ckpt, timestep) in enumerate(
                    zip(payload["ckpt_indices"], payload["timesteps"])
                )
            }
            for payload in query_payloads
        ]
        query_features = [
            np.asarray(payload["query_features"], dtype=np.float32)
            for payload in query_payloads
        ]
        del query_payloads

        path = train_shard(root, args, component, start, end)
        print(f"[temporal-thirds] loading {component.label}: {path}", flush=True)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpt_indices = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            positions = np.asarray(payload["snapshot_positions"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)

        if score_indices is None:
            score_indices = indices
            for segment in SEGMENTS:
                for variant in VARIANTS:
                    scores[(segment, variant)] = np.zeros(
                        (num_queries, len(indices)), dtype=np.float64
                    )
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")

        print(
            f"[temporal-thirds] {component.label} loaded | shape={train.shape}",
            flush=True,
        )
        for term_i, (ckpt, timestep, position) in enumerate(
            zip(ckpt_indices, timesteps, positions)
        ):
            segment = position_to_segment.get(int(position))
            if segment is None:
                raise ValueError(
                    f"unexpected snapshot position {int(position)} in {path}"
                )
            query_rows = []
            query_indices = [
                query_map.get((int(ckpt), int(timestep)))
                for query_map in query_maps
            ]
            if all(query_i is None for query_i in query_indices):
                if int(ckpt) == 0:
                    skipped_terms.add((component.label, int(position), int(timestep)))
                continue
            if any(query_i is None for query_i in query_indices):
                raise ValueError(
                    f"query artifacts disagree on checkpoint/timestep {(int(ckpt), int(timestep))}"
                )
            for spec_i, query_map in enumerate(query_maps):
                query_i = query_indices[spec_i]
                query_rows.append(query_features[spec_i][query_i])
            query_matrix = np.stack(query_rows, axis=0).astype(np.float32, copy=False)
            query_norms = np.sqrt(
                np.einsum("ij,ij->i", query_matrix, query_matrix, optimize=True)
            )
            query_normalized = query_matrix / np.maximum(query_norms, query_eps)[:, None]

            train_term = np.asarray(train[term_i], dtype=np.float32)
            train_norms = np.sqrt(
                np.einsum("ij,ij->i", train_term, train_term, optimize=True)
            )
            train_denominator = np.maximum(train_norms, train_eps)
            raw = train_term @ query_matrix.T
            query_l2 = train_term @ query_normalized.T
            weight = float(weights[term_i])

            scores[(segment, "raw")] += (weight * raw).T
            scores[(segment, "query_l2")] += (weight * query_l2).T
            scores[(segment, "train_l2")] += (
                weight * raw / train_denominator[:, None]
            ).T
            scores[(segment, "query_train_l2")] += (
                weight * query_l2 / train_denominator[:, None]
            ).T
            count_key = (int(ckpt), segment)
            segment_term_counts[count_key] = segment_term_counts.get(count_key, 0) + 1
            if (term_i + 1) % 50 == 0 or term_i + 1 == len(train):
                print(
                    f"[temporal-thirds] {component.label} term {term_i + 1}/{len(train)}",
                    flush=True,
                )
        del train, query_maps, query_features
        gc.collect()

    expected_counts = {
        segment: segment_term_counts.get((0, segment), 0) for segment in SEGMENTS
    }
    bad_counts = {}
    for ckpt in range(49):
        for segment in SEGMENTS:
            count = segment_term_counts.get((ckpt, segment), 0)
            if count != expected_counts[segment]:
                bad_counts[(ckpt, segment)] = count
    if any(count == 0 for count in expected_counts.values()) or bad_counts:
        preview = list(sorted(bad_counts.items()))[:10]
        raise ValueError(
            f"inconsistent checkpoint/segment term counts; "
            f"expected={expected_counts} bad={preview}"
        )
    print(
        f"[temporal-thirds] effective terms per checkpoint: {expected_counts}",
        flush=True,
    )
    if skipped_terms:
        print(
            "[temporal-thirds] train terms absent from existing query artifacts "
            f"(checkpoint 1 shown): {sorted(skipped_terms)}",
            flush=True,
        )
    if score_indices is None:
        raise RuntimeError("no train artifacts were loaded")

    for spec_i, spec in enumerate(specs):
        for segment in SEGMENTS:
            for variant in VARIANTS:
                out_dir = score_shard_dir(
                    root, args, spec, segment, variant, start, end
                )
                write_scores(
                    out_dir,
                    scores[(segment, variant)][spec_i],
                    score_indices,
                    {
                        "mode": "aligned_temporal_thirds",
                        "segment": segment,
                        "variant": variant,
                        "effective_terms_per_checkpoint": expected_counts,
                        "train_namespaces": [
                            args.base_train_namespace,
                            args.addon_a_train_namespace,
                            args.addon_b_train_namespace,
                        ],
                        "query_namespaces": {
                            component.label: component.query_namespace
                            for component in components
                        },
                    },
                )
    print(f"[temporal-thirds] worker done range={start}-{end}", flush=True)


def merge_scores(repo_root: Path, root: Path, args: argparse.Namespace) -> None:
    merge_script = root.parent / "common" / "merge_score_shards.py"
    ranges = parse_ranges(args.score_index_ranges, size=args.size)
    for spec in build_query_specs(args):
        for segment in SEGMENTS:
            for variant in VARIANTS:
                output_dir = score_final_dir(root, args, spec, segment, variant)
                shards = [
                    score_shard_dir(root, args, spec, segment, variant, start, end)
                    for start, end in ranges
                ]
                subprocess.run(
                    [sys.executable, str(merge_script), "--output-dir", str(output_dir), *map(str, shards)],
                    cwd=repo_root,
                    check=True,
                )


def run_lds(repo_root: Path, root: Path, args: argparse.Namespace) -> None:
    evaluator = root.parent / "common" / "fast_lds_score_eval.py"
    dataset_config = root / "dataset_config.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent)
    for spec in build_query_specs(args):
        query = f"query_{query_tag(str(spec['query']))}"
        seed = f"initial_seed_{int(spec['initial_seed'])}"
        target_root = (
            root / "result" / args.experiment / "eval" / "prompted_solo"
            / query / seed / args.base_train_namespace / "lds" / "traj_tracin"
        )
        target_csvs = sorted(target_root.glob("*/lds_results.csv"))
        if not target_csvs:
            raise FileNotFoundError(f"missing cached LDS targets: {target_root}")
        eval_root = (
            root / "result" / args.experiment / "eval" / "prompted_solo"
            / query / seed / args.output_namespace / "lds"
        )
        for segment in SEGMENTS:
            for variant in VARIANTS:
                algorithm = f"traj_tracin_{segment}_{variant}"
                score_dir = score_final_dir(root, args, spec, segment, variant)
                for target_csv in target_csvs:
                    out_dir = eval_root / algorithm / target_csv.parent.name
                    subprocess.run(
                        [
                            sys.executable,
                            str(evaluator),
                            str(dataset_config),
                            "--target-results", str(target_csv),
                            "--score-file", str(score_dir),
                            "--algorithm", algorithm,
                            "--target-function", target_csv.parent.name,
                            "--prediction-subset", "kept",
                            "--prediction-sign", "-1",
                            "--out-dir", str(out_dir),
                        ],
                        cwd=repo_root,
                        env=env,
                        check=True,
                    )


def print_summary(root: Path, args: argparse.Namespace) -> None:
    print("\nMean LDS by temporal third:")
    print(f"{'segment':14s} {'variant':16s} {'mean':>8s} {'n':>4s}")
    print("-" * 46)
    for segment in SEGMENTS:
        for variant in VARIANTS:
            algorithm = f"traj_tracin_{segment}_{variant}"
            values = []
            for path in (root / "result" / args.experiment / "eval").glob(
                f"prompted_solo/query_*/initial_seed_*/{args.output_namespace}/"
                f"lds/{algorithm}/*/lds_summary.json"
            ):
                values.append(float(json.loads(path.read_text())["lds_spearman"]))
            mean = sum(values) / len(values) if values else float("nan")
            print(f"{segment:14s} {variant:16s} {mean:8.3f} {len(values):4d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare initial, middle, and end temporal thirds of CIFAR5 TrajTracIn."
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
    parser.add_argument("--base-train-namespace", default="raw_nextckpt_school_traj_aligned_10x10")
    parser.add_argument("--addon-a-train-namespace", default="raw_nextckpt_school_traj_addon_mid10")
    parser.add_argument("--addon-b-train-namespace", default="raw_nextckpt_school_traj_addon_mid10_b")
    parser.add_argument(
        "--base-query-namespace",
        "--query-namespace",
        dest="base_query_namespace",
        default="raw_nextckpt_school_traj_10x10",
    )
    parser.add_argument(
        "--addon-a-query-namespace",
        default="raw_nextckpt_school_traj_addon_mid10_exact10",
    )
    parser.add_argument(
        "--addon-b-query-namespace",
        default="raw_nextckpt_school_traj_addon_mid10_b_exact10",
    )
    parser.add_argument("--output-namespace", default="raw_nextckpt_school_traj_temporal_thirds_30term")
    parser.add_argument("--score-index-ranges", default="1-2500,2501-5000,5001-7500,7501-10000")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default="local")
    parser.add_argument("--use-task-affinity", action="store_true")
    parser.add_argument("--query-normalize-eps", type=float, default=1e-8)
    parser.add_argument("--train-normalize-eps", type=float, default=1e-8)
    parser.add_argument("--worker-range", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent.parent
    if args.worker_range:
        start, end = parse_ranges(args.worker_range, size=args.size)[0]
        run_worker(root, args, start, end)
        return

    positions = sorted(segment_position_map())
    print("Temporal thirds (trajectory position 0 is initial/noisy):", flush=True)
    for i, segment in enumerate(SEGMENTS):
        current = positions[i * 10 : (i + 1) * 10]
        print(f"  {segment}: positions={current}", flush=True)
    if not args.execute:
        print("[dry-run] add --execute to calculate scores and LDS", flush=True)
        return

    ranges = parse_ranges(args.score_index_ranges, size=args.size)
    gpus = parse_gpus(args)
    worker_gpu_ids = worker_gpus(args, gpus)
    jobs = []
    for range_i, (start, end) in enumerate(ranges):
        slot = slot_for(range_i, len(worker_gpu_ids))
        jobs.append(
            Job(
                name=f"cifar5_temporal_thirds_{start}_{end}",
                cmd=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-range", f"{start}-{end}"],
                cwd=repo_root,
                env=gpu_env(os.environ.copy(), worker_gpu_ids[slot]),
                log_path=(
                    root / "result" / args.experiment / "logs" / "traj_temporal_thirds"
                    / args.output_namespace / f"range_{start}_{end}_gpu_{worker_gpu_ids[slot]}.log"
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
    merge_scores(repo_root, root, args)
    run_lds(repo_root, root, args)
    print_summary(root, args)
    print("[done] temporal-third score and LDS comparison complete", flush=True)


if __name__ == "__main__":
    main()
