#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from run_cifar5_multi_experiment import Job, gpu_env, parse_gpus, run_parallel_jobs, slot_for, worker_gpus
from run_cifar5_multi_random_prompted_queries import build_query_specs, query_tag


TRAIN_ARTIFACT = "train_datapoint_gradient_artifact.npz"
QUERY_ARTIFACT = "query_gradient_artifact.npz"
VARIANTS = {
    "raw": ("score", "traj_tracin"),
    "query_l2": ("score_query_normalized", "traj_tracin_query_normalized"),
    "train_l2": ("score_train_l2_normalized", "traj_tracin_train_l2_normalized"),
    "query_train_l2": (
        "score_query_train_l2_normalized",
        "traj_tracin_query_train_l2_normalized",
    ),
}


@dataclass(frozen=True)
class Component:
    label: str
    train_namespace: str
    query_namespace: str
    score_namespace: str


def parse_ranges(text: str, *, size: int) -> list[tuple[int, int]]:
    ranges = []
    for item in text.replace(";", ",").replace(" ", ",").split(","):
        if not item.strip():
            continue
        start_text, end_text = item.strip().replace(":", "-").split("-", 1)
        start, end = int(start_text), int(end_text)
        if start < 1 or end < start or end > size:
            raise ValueError(f"invalid range {item!r}; expected bounds within 1-{size}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("no score ranges supplied")
    return ranges


def train_shard(root: Path, args: argparse.Namespace, component: Component, start: int, end: int) -> Path:
    return (
        root
        / "result"
        / args.experiment
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient_{component.train_namespace}"
        / "traj_tracin"
        / "datapoint_shards"
        / f"range_{start}_{end}"
        / TRAIN_ARTIFACT
    )


def query_artifact(root: Path, args: argparse.Namespace, spec: dict[str, int | str], namespace: str) -> Path:
    query = str(spec["query"])
    seed = int(spec["initial_seed"])
    return (
        root
        / "result"
        / args.experiment
        / "sample"
        / "cifar"
        / f"prompt_{query_tag(query)}"
        / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
        / f"seed_{seed:06d}_query_gradient_{namespace}"
        / "traj_tracin"
        / QUERY_ARTIFACT
    )


def score_root(root: Path, args: argparse.Namespace, spec: dict[str, int | str], namespace: str) -> Path:
    return (
        root
        / "result"
        / args.experiment
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
        / f"query_{query_tag(str(spec['query']))}"
        / f"initial_seed_{int(spec['initial_seed'])}"
        / namespace
        / "traj_tracin"
    )


def variant_dir(base: Path, variant: str) -> Path:
    return base / VARIANTS[variant][0]


def shard_dir(base: Path, variant: str, start: int, end: int) -> Path:
    return variant_dir(base, variant) / "datapoint_shards" / f"range_{start}_{end}"


def score_complete(path: Path) -> bool:
    return (path / "scores.npy").is_file() and (path / "score_indices.npy").is_file()


def ddim_step_squared_mass(timesteps: np.ndarray, args: argparse.Namespace) -> float:
    betas = np.linspace(args.beta_start, args.beta_end, args.timesteps_total, dtype=np.float64)
    alpha_bars = np.cumprod(1.0 - betas)
    mass = 0.0
    for timestep in np.unique(np.asarray(timesteps, dtype=np.int64)):
        alpha_bar_t = alpha_bars[timestep]
        alpha_bar_prev = alpha_bars[timestep - 1] if timestep > 0 else 1.0
        coefficient = (
            np.sqrt(1.0 - alpha_bar_prev)
            - np.sqrt(alpha_bar_prev / alpha_bar_t) * np.sqrt(1.0 - alpha_bar_t)
        )
        mass += float(coefficient * coefficient)
    if mass <= 0.0:
        raise ValueError("selected component has no positive DDIM step-squared mass")
    return mass


def component_score_mass(
    root: Path,
    args: argparse.Namespace,
    component: Component,
) -> float:
    if args.timestep_weighting == "uniform":
        return 1.0
    start, end = parse_ranges(args.score_index_ranges, size=args.size)[0]
    path = train_shard(root, args, component, start, end)
    with np.load(path, allow_pickle=False) as payload:
        timesteps = np.asarray(payload["timesteps"], dtype=np.int64)
    if args.timestep_region != "all":
        allowed = set(args.resolved_timestep_allowlist)
        timesteps = np.asarray([value for value in timesteps if int(value) in allowed])
    return ddim_step_squared_mass(timesteps, args)


def write_scores(out_dir: Path, scores: np.ndarray, indices: np.ndarray, metadata: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "scores.npy", np.asarray(scores, dtype=np.float64))
    np.save(out_dir / "score_indices.npy", np.asarray(indices, dtype=np.int64))
    order = np.argsort(-scores)
    top = [
        {
            "rank": rank,
            "idx": int(indices[i]),
            "idx_1based": int(indices[i]) + 1,
            "score": float(scores[i]),
        }
        for rank, i in enumerate(order[: min(2000, len(order))], start=1)
    ]
    (out_dir / "top_scores.json").write_text(json.dumps({"top": top, "num_scored": len(scores)}, indent=2))
    (out_dir / "score_artifact_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def merge_component_scores(
    repo_root: Path,
    root: Path,
    args: argparse.Namespace,
    specs: list[dict[str, int | str]],
    component: Component,
    ranges: list[tuple[int, int]],
) -> None:
    merge_script = root.parent / "common" / "merge_score_shards.py"
    for spec in specs:
        base = score_root(root, args, spec, component.score_namespace)
        for variant in VARIANTS:
            final_dir = variant_dir(base, variant)
            if score_complete(final_dir):
                continue
            shards = [shard_dir(base, variant, start, end) for start, end in ranges]
            missing = [path for path in shards if not score_complete(path)]
            if missing:
                raise FileNotFoundError(f"missing {component.label}/{variant} score shard: {missing[0]}")
            subprocess.run(
                [sys.executable, str(merge_script), "--output-dir", str(final_dir), *map(str, shards)],
                cwd=repo_root,
                check=True,
            )


def combine_components(
    root: Path,
    args: argparse.Namespace,
    specs: list[dict[str, int | str]],
    components: dict[str, Component],
    combinations: dict[str, tuple[str, ...]],
) -> None:
    component_masses = {
        label: component_score_mass(root, args, component)
        for label, component in components.items()
    }
    for output_namespace, labels in combinations.items():
        for spec in specs:
            output_base = score_root(root, args, spec, output_namespace)
            for variant in VARIANTS:
                out_dir = variant_dir(output_base, variant)
                if score_complete(out_dir):
                    continue
                inputs = [
                    variant_dir(score_root(root, args, spec, components[label].score_namespace), variant)
                    for label in labels
                ]
                payloads = [
                    (
                        np.asarray(np.load(path / "scores.npy"), dtype=np.float64),
                        np.asarray(np.load(path / "score_indices.npy"), dtype=np.int64),
                    )
                    for path in inputs
                ]
                indices = payloads[0][1]
                if any(not np.array_equal(indices, item_indices) for _, item_indices in payloads[1:]):
                    raise ValueError(f"score indices differ for combination {output_namespace}/{variant}")
                masses = np.asarray([component_masses[label] for label in labels], dtype=np.float64)
                scores = np.average(
                    np.stack([item_scores for item_scores, _ in payloads]),
                    axis=0,
                    weights=masses,
                )
                write_scores(
                    out_dir,
                    scores,
                    indices,
                    {
                        "mode": (
                            "ddim_step_squared_global_average"
                            if args.timestep_weighting == "ddim_step_squared"
                            else "equal_term_component_average"
                        ),
                        "timestep_weighting": args.timestep_weighting,
                        "component_masses": {
                            label: float(mass) for label, mass in zip(labels, masses)
                        },
                        "variant": variant,
                        "components": labels,
                        "component_score_dirs": [str(path) for path in inputs],
                        "num_components": len(labels),
                        "num_scores": len(scores),
                    },
                )
                print(f"[combine] {output_namespace} {variant} query={spec['query']} seed={spec['initial_seed']}", flush=True)


def run_fast_lds(
    repo_root: Path,
    root: Path,
    args: argparse.Namespace,
    specs: list[dict[str, int | str]],
    score_namespaces: list[str],
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent)
    env["EXPERIMENT_TAG"] = args.experiment
    env["TRAIN_SEED"] = str(args.train_seed)
    evaluator = root.parent / "common" / "fast_lds_score_eval.py"
    dataset_config = root / "dataset_config.py"
    for spec in specs:
        query_component = f"query_{query_tag(str(spec['query']))}"
        seed_component = f"initial_seed_{int(spec['initial_seed'])}"
        target_root = (
            root
            / "result"
            / args.experiment
            / "eval"
            / "prompted_solo"
            / query_component
            / seed_component
            / (args.lds_target_namespace or args.base_train_namespace)
            / "lds"
            / args.lds_target_algorithm
        )
        target_csvs = sorted(target_root.glob("*/lds_results.csv"))
        if not target_csvs:
            print(f"[LDS missing target cache] {query_component} {seed_component}: {target_root}", flush=True)
            continue
        for namespace in score_namespaces:
            score_base = score_root(root, args, spec, namespace)
            eval_base = (
                root
                / "result"
                / args.experiment
                / "eval"
                / "prompted_solo"
                / query_component
                / seed_component
                / namespace
                / "lds"
            )
            for target_csv in target_csvs:
                target = target_csv.parent.name
                for variant, (_score_component, algorithm) in VARIANTS.items():
                    score_dir = variant_dir(score_base, variant)
                    out_dir = eval_base / algorithm / target
                    if (out_dir / "lds_summary.json").is_file():
                        continue
                    subprocess.run(
                        [
                            sys.executable,
                            str(evaluator),
                            str(dataset_config),
                            "--target-results",
                            str(target_csv),
                            "--score-file",
                            str(score_dir),
                            "--algorithm",
                            algorithm,
                            "--target-function",
                            target,
                            "--prediction-subset",
                            "kept",
                            "--prediction-sign",
                            "-1",
                            "--out-dir",
                            str(out_dir),
                        ],
                        cwd=repo_root,
                        env=env,
                        check=True,
                    )


def print_lds_summary(
    root: Path,
    args: argparse.Namespace,
    specs: list[dict[str, int | str]],
    score_namespaces: list[str],
) -> None:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for namespace in score_namespaces:
        for spec in specs:
            eval_base = (
                root
                / "result"
                / args.experiment
                / "eval"
                / "prompted_solo"
                / f"query_{query_tag(str(spec['query']))}"
                / f"initial_seed_{int(spec['initial_seed'])}"
                / namespace
                / "lds"
            )
            for variant, (_score_component, algorithm) in VARIANTS.items():
                for summary_path in (eval_base / algorithm).glob("*/lds_summary.json"):
                    payload = json.loads(summary_path.read_text())
                    values[(namespace, variant, summary_path.parent.name)].append(float(payload["lds_spearman"]))

    targets = ("endpoint_counterfactual", "noise_trajectory", "traj_counterfactual", "simple_loss")
    aliases = {
        "endpoint_counterfactual": ("endpoint_counterfactual", "endpoint_contarfactual"),
        "noise_trajectory": ("noise_trajectory",),
        "traj_counterfactual": ("traj_counterfactual", "traj_contarfactual"),
        "simple_loss": ("simple_loss",),
    }
    print("\nMean LDS by score namespace and normalization:", flush=True)
    print(f"{'namespace':52s} {'variant':15s} {'end':>7s} {'noise':>7s} {'traj':>7s} {'simple':>7s} {'all':>7s} {'n':>4s}")
    print("-" * 114)
    for namespace in score_namespaces:
        for variant in VARIANTS:
            target_values = []
            means = []
            for target in targets:
                current = []
                for alias in aliases[target]:
                    current.extend(values[(namespace, variant, alias)])
                means.append(sum(current) / len(current) if current else float("nan"))
                target_values.extend(current)
            overall = sum(target_values) / len(target_values) if target_values else float("nan")
            print(
                f"{namespace:52s} {variant:15s}"
                f" {means[0]:7.3f} {means[1]:7.3f} {means[2]:7.3f} {means[3]:7.3f}"
                f" {overall:7.3f} {len(target_values):4d}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch CIFAR5 TrajTracIn raw/Q/L/Q+L scoring for base and add-ons.")
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
    parser.add_argument("--base-query-namespace", default="raw_nextckpt_school_traj_10x10")
    parser.add_argument(
        "--lds-target-namespace",
        default="",
        help="Namespace containing reusable LDS result CSVs; defaults to --base-train-namespace.",
    )
    parser.add_argument(
        "--lds-target-algorithm",
        default="traj_tracin",
        help="LDS subdirectory supplying target CSVs, e.g. das_lambda_500.",
    )
    parser.add_argument(
        "--base-score-namespace",
        default="",
        help="Output namespace for base scores; defaults to --base-train-namespace.",
    )
    parser.add_argument("--addon-a-namespace", default="raw_nextckpt_school_traj_addon_mid10")
    parser.add_argument("--addon-b-namespace", default="raw_nextckpt_school_traj_addon_mid10_b")
    parser.add_argument("--addon-c-namespace", default="raw_nextckpt_school_traj_addon_mid10_c")
    for label in ("a", "b", "c"):
        parser.add_argument(f"--addon-{label}-train-namespace", default="")
        parser.add_argument(f"--addon-{label}-query-namespace", default="")
        parser.add_argument(f"--addon-{label}-score-namespace", default="")
    parser.add_argument("--combined-a-namespace", default="raw_nextckpt_school_traj_combined20_mid10_a")
    parser.add_argument("--combined-b-namespace", default="raw_nextckpt_school_traj_combined20_mid10_b")
    parser.add_argument("--combined-c-namespace", default="raw_nextckpt_school_traj_combined20_mid10_c")
    parser.add_argument("--combined-ab-namespace", default="raw_nextckpt_school_traj_combined30_mid10_ab")
    parser.add_argument("--combined-abc-namespace", default="raw_nextckpt_school_traj_combined40_mid10_abc")
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
    parser.add_argument(
        "--timestep-weighting",
        choices=("uniform", "ddim_step_squared"),
        default="uniform",
        help="Weight timesteps uniformly or by the squared deterministic DDIM step coefficient.",
    )
    parser.add_argument("--timesteps-total", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=0.0001)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument(
        "--timestep-region",
        choices=("all", "initial", "end"),
        default="all",
        help="Restrict scoring to the largest-t (initial) or smallest-t (end) timesteps.",
    )
    parser.add_argument("--region-terms", type=int, default=10)
    parser.add_argument("--skip-component-score", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument(
        "--only-base",
        action="store_true",
        help="Score only the base train/query pair and do not require add-on artifacts.",
    )
    parser.add_argument(
        "--components",
        default="base,addon_a,addon_b,addon_c",
        help="Comma-separated components to score (base, addon_a, addon_b, addon_c).",
    )
    parser.add_argument(
        "--score-components",
        default="",
        help="Only calculate these selected components; defaults to all --components.",
    )
    parser.add_argument(
        "--combinations",
        default="",
        help="Only build these combinations: a,b,c,ab,abc. Defaults to all available.",
    )
    parser.add_argument("--fast-lds", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent.parent
    specs = build_query_specs(args)
    ranges = parse_ranges(args.score_index_ranges, size=args.size)
    component_defaults = {
        "base": Component(
            "base",
            args.base_train_namespace,
            args.base_query_namespace,
            args.base_score_namespace or args.base_train_namespace,
        ),
        "addon_a": Component(
            "addon_a",
            args.addon_a_train_namespace or args.addon_a_namespace,
            args.addon_a_query_namespace or args.addon_a_namespace,
            args.addon_a_score_namespace or args.addon_a_namespace,
        ),
        "addon_b": Component(
            "addon_b",
            args.addon_b_train_namespace or args.addon_b_namespace,
            args.addon_b_query_namespace or args.addon_b_namespace,
            args.addon_b_score_namespace or args.addon_b_namespace,
        ),
        "addon_c": Component(
            "addon_c",
            args.addon_c_train_namespace or args.addon_c_namespace,
            args.addon_c_query_namespace or args.addon_c_namespace,
            args.addon_c_score_namespace or args.addon_c_namespace,
        ),
    }
    combination_defaults = {
        "a": (args.combined_a_namespace, ("base", "addon_a")),
        "b": (args.combined_b_namespace, ("base", "addon_b")),
        "c": (args.combined_c_namespace, ("base", "addon_c")),
        "ab": (args.combined_ab_namespace, ("base", "addon_a", "addon_b")),
        "abc": (args.combined_abc_namespace, ("base", "addon_a", "addon_b", "addon_c")),
    }
    if args.only_base:
        selected_components = ("base",)
    else:
        selected_components = tuple(item.strip() for item in args.components.split(",") if item.strip())
    unknown_components = sorted(set(selected_components) - set(component_defaults))
    if unknown_components:
        parser.error(f"unknown --components: {','.join(unknown_components)}")
    components = {label: component_defaults[label] for label in selected_components}

    if args.score_components:
        score_component_labels = tuple(
            item.strip() for item in args.score_components.split(",") if item.strip()
        )
    else:
        score_component_labels = selected_components
    unknown_score_components = sorted(set(score_component_labels) - set(components))
    if unknown_score_components:
        parser.error(
            "--score-components must be included in --components: "
            + ",".join(unknown_score_components)
        )

    if args.combinations:
        selected_combinations = tuple(
            item.strip() for item in args.combinations.split(",") if item.strip()
        )
        unknown_combinations = sorted(set(selected_combinations) - set(combination_defaults))
        if unknown_combinations:
            parser.error(f"unknown --combinations: {','.join(unknown_combinations)}")
    else:
        selected_combinations = tuple(combination_defaults)
    combinations = {
        namespace: labels
        for key in selected_combinations
        for namespace, labels in (combination_defaults[key],)
        if all(label in components for label in labels)
    }
    if args.only_base:
        combinations = {}
    timestep_allowlist: tuple[int, ...] = ()
    if args.timestep_region != "all":
        all_timesteps = set()
        first_start, first_end = ranges[0]
        for component in components.values():
            path = train_shard(root, args, component, first_start, first_end)
            if not path.is_file():
                raise FileNotFoundError(str(path))
            with np.load(path, allow_pickle=False) as payload:
                all_timesteps.update(int(value) for value in np.asarray(payload["timesteps"]).reshape(-1))
        ordered_timesteps = sorted(all_timesteps)
        if args.region_terms < 1 or args.region_terms > len(ordered_timesteps):
            parser.error(
                f"--region-terms must be within 1-{len(ordered_timesteps)} for selected components"
            )
        if args.timestep_region == "initial":
            timestep_allowlist = tuple(ordered_timesteps[-args.region_terms :])
        else:
            timestep_allowlist = tuple(ordered_timesteps[: args.region_terms])
        print(
            f"timestep_region={args.timestep_region} terms={len(timestep_allowlist)} "
            f"allowlist={timestep_allowlist}",
            flush=True,
        )
    args.resolved_timestep_allowlist = timestep_allowlist
    all_score_namespaces = [
        components[label].score_namespace for label in score_component_labels
    ] + list(combinations)

    print(f"queries={len(specs)} components={list(components)} ranges={ranges}", flush=True)
    print("variants=raw,query_l2,train_l2,query_train_l2", flush=True)
    if not args.execute:
        print("[dry-run] pass --execute to write scores", flush=True)
        return

    if not args.skip_component_score:
        gpus = parse_gpus(args)
        worker_gpu_ids = worker_gpus(args, gpus)
        max_parallel = max(1, min(args.max_parallel, len(worker_gpu_ids)))
        for component_label in score_component_labels:
            component = components[component_label]
            jobs = []
            for range_i, (start, end) in enumerate(ranges):
                train_path = train_shard(root, args, component, start, end)
                if not train_path.is_file():
                    raise FileNotFoundError(str(train_path))
                batch_jobs = []
                for spec in specs:
                    output_base = score_root(root, args, spec, component.score_namespace)
                    if all(score_complete(shard_dir(output_base, variant, start, end)) for variant in VARIANTS):
                        continue
                    query_path = query_artifact(root, args, spec, component.query_namespace)
                    if not query_path.is_file():
                        raise FileNotFoundError(str(query_path))
                    batch_jobs.append(
                        {
                            "query_path": str(query_path),
                            "output_dir": str(shard_dir(output_base, "raw", start, end)),
                            "label": f"{spec['query']} seed={spec['initial_seed']}",
                        }
                    )
                if not batch_jobs:
                    continue
                slot = slot_for(range_i, len(worker_gpu_ids))
                env = os.environ.copy()
                env.update(
                    {
                        "PYTHONUNBUFFERED": "1",
                        "TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP": "1",
                        "TRACIN_SCORE_QUERY_NORMALIZE": "1",
                        "TRACIN_SCORE_TRAIN_NORMALIZE": "1",
                        "TRACIN_SCORE_QUERY_NORMALIZE_EPS": str(args.query_normalize_eps),
                        "TRACIN_SCORE_TRAIN_NORMALIZE_EPS": str(args.train_normalize_eps),
                        "TRACIN_SCORE_TIMESTEP_ALLOWLIST": ",".join(
                            str(value) for value in timestep_allowlist
                        ),
                        "TRACIN_SCORE_TIMESTEP_WEIGHTING": args.timestep_weighting,
                        "TRACIN_SCORE_TIMESTEPS_TOTAL": str(args.timesteps_total),
                        "TRACIN_SCORE_BETA_START": str(args.beta_start),
                        "TRACIN_SCORE_BETA_END": str(args.beta_end),
                        "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(train_path),
                        "TRACIN_SCORE_BATCH_JOBS": json.dumps(batch_jobs),
                    }
                )
                jobs.append(
                    Job(
                        name=f"cifar5_norm_{component.label}_{start}_{end}",
                        cmd=[sys.executable, "03_score_batch.py"],
                        cwd=root / "data_attribution" / "traj_tracin",
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=(
                            root
                            / "result"
                            / args.experiment
                            / "logs"
                            / "traj_tracin_norm_sweep"
                            / component.label
                            / f"range_{start}_{end}_gpu_{worker_gpu_ids[slot]}.log"
                        ),
                        slot=slot,
                    )
                )
            run_parallel_jobs(jobs, args=args, execute=True, max_parallel=max_parallel)
            merge_component_scores(repo_root, root, args, specs, component, ranges)

    if not args.skip_combine:
        combine_components(root, args, specs, components, combinations)

    if args.fast_lds:
        run_fast_lds(repo_root, root, args, specs, all_score_namespaces)
        print_lds_summary(root, args, specs, all_score_namespaces)

    print("[done] CIFAR5 TrajTracIn norm sweep complete", flush=True)


if __name__ == "__main__":
    main()
