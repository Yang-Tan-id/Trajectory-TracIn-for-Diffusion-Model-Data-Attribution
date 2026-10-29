#!/usr/bin/env python3
"""Export five-method per-query LDS results for CelebA or Shapes3D."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
    "predicted_noise_kl",
    "simple_loss_50x1",
    "vlb_nll",
)
@dataclass(frozen=True)
class DatasetConfig:
    title: str
    relative_root: str
    experiment: str
    query_file: str
    prompt_tag_limit: int
    empty_tag: str


@dataclass(frozen=True)
class MethodConfig:
    label: str
    variant: str
    raw_namespace: str
    adamw_namespace: str
    raw_sign: str
    adamw_sign: str
    raw_multiplier: float
    adamw_multiplier: float


CONFIGS = {
    "celeba": DatasetConfig(
        title="CELEBA",
        relative_root="diffusion_jax_refined/celeba",
        experiment="attr40_experiment1",
        query_file="queries_attr40_val25_x4.json",
        prompt_tag_limit=150,
        empty_tag="empty_prompt",
    ),
    "shape3d": DatasetConfig(
        title="SHAPES3D",
        relative_root="diffusion_jax_refined/3dshapes",
        experiment="experiment1",
        query_file="queries_in_distribution_plus_zero_seed_100_219.json",
        prompt_tag_limit=160,
        empty_tag="empty",
    ),
}


def method_configs(dataset: str) -> tuple[MethodConfig, ...]:
    if dataset == "shape3d":
        pint_raw = (
            "traj_tracin_recreate_raw_polluted_endpoint_delta_l2normalized_"
            "timestamp_sum_squared_q0_99"
        )
        pint_adamw = (
            "traj_tracin_recreate_adamw_full_polluted_endpoint_"
            "delta_l2normalized_timestamp_sum_squared_q0_99"
        )
        tracin_raw = "traj_tracin_endpoint_tracin_train100x1_query100x1_q0_99"
        tracin_adamw = (
            "traj_tracin_endpoint_tracin_adamw_full_train100x1_query100x1_q0_99"
        )
        retrac_raw = "traj_tracin_paper_retrac_exact4_endpoint100x1_q0_99"
        retrac_adamw = (
            "traj_tracin_paper_retrac_adamw_full_exact4_endpoint100x1_q0_99"
        )
        pint_signs = ("p1", "p1")
        baseline_signs = ("m1", "m1")
        pint_multipliers = (-1.0, -1.0)
    else:
        pint_raw = (
            "traj_tracin_celeba_endpoint_pollute_raw_gradient_train1x50_"
            "query1x50_timestamp_sum_squared_shape3d_semantics"
        )
        pint_adamw = (
            "traj_tracin_celeba_endpoint_pollute_adamw_full_train1x50_"
            "query1x50_timestamp_sum_squared_shape3d_semantics"
        )
        tracin_raw = (
            "traj_tracin_celeba_endpoint_tracin_raw_gradient_train_mean50_"
            "query_mean50_linear_shape3d_semantics"
        )
        tracin_adamw = (
            "traj_tracin_celeba_endpoint_tracin_adamw_full_train_mean50_"
            "query_mean50_linear_shape3d_semantics"
        )
        retrac_raw = (
            "traj_tracin_celeba_paper_retrac_raw_gradient_train_replay4_batch1_"
            "query_mean50_of_full_l2_linear_shape3d_semantics"
        )
        retrac_adamw = (
            "traj_tracin_celeba_paper_retrac_adamw_full_train_replay4_batch1_"
            "query_mean50_of_full_l2_linear_shape3d_semantics"
        )
        # The completed CelebA LDS cache used the raw-gradient descent
        # orientation (m1) and the stored AdamW update orientation (p1).
        pint_signs = ("m1", "p1")
        baseline_signs = ("m1", "p1")
        pint_multipliers = (1.0, 1.0)
    return (
        MethodConfig(
            "PINT / RAW",
            "raw",
            pint_raw,
            pint_adamw,
            *pint_signs,
            *pint_multipliers,
        ),
        MethodConfig(
            "PINT / BOTH-L2",
            "query_train_l2",
            pint_raw,
            pint_adamw,
            *pint_signs,
            *pint_multipliers,
        ),
        MethodConfig(
            "TRACIN / RAW",
            "raw",
            tracin_raw,
            tracin_adamw,
            *baseline_signs,
            1.0,
            -1.0,
        ),
        MethodConfig(
            "TRACIN / BOTH-L2",
            "query_train_l2",
            tracin_raw,
            tracin_adamw,
            *baseline_signs,
            1.0,
            -1.0,
        ),
        MethodConfig(
            "RETRAC / BOTH-L2",
            "query_train_l2",
            retrac_raw,
            retrac_adamw,
            *baseline_signs,
            1.0,
            -1.0,
        ),
    )


def prompt_tag(prompt: str, config: DatasetConfig) -> str:
    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[: config.prompt_tag_limit] or config.empty_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--expected-subsets", type=int, default=192)
    parser.add_argument("--query-ids", default="0-99")
    parser.add_argument(
        "--targets",
        default="endpoint_contarfactual,traj_contarfactual,simple_loss,noise_trajectory",
        help="Comma/space-separated LDS targets to export.",
    )
    parser.add_argument(
        "--transforms",
        choices=("both", "raw-gradient", "adamw"),
        default="both",
        help="Export both transforms or only one transform.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_query_ids(text: str) -> list[int]:
    values: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise ValueError("query IDs must be nonempty and unique")
    return values


def load_summary(
    path: Path, expected_subsets: int, sign: str
) -> tuple[dict[str, object], Path]:
    """Load the preferred prediction sign, or its algebraic inverse if absent."""
    attempts: list[tuple[str, list[Path], list[tuple[dict[str, object], Path]]]] = []
    opposite = {"p1": "m1", "m1": "p1"}[sign]
    for candidate_sign in (sign, opposite):
        candidates = sorted(
            (path / f"pred_kept_sign_{candidate_sign}").glob("*/lds_summary.json")
        )
        decoded: list[tuple[dict[str, object], Path]] = []
        for candidate in candidates:
            payload = json.loads(candidate.read_text())
            if int(payload.get("num_models", -1)) == expected_subsets:
                decoded.append((payload, candidate))
        attempts.append((candidate_sign, candidates, decoded))
        if len(decoded) == 1:
            return decoded[0]
        if len(decoded) > 1:
            break
    details = "; ".join(
        f"sign={candidate_sign}: {len(decoded)} of {len(candidates)} candidates "
        f"{candidates}"
        for candidate_sign, candidates, decoded in attempts
    )
    raise RuntimeError(
        f"expected one {expected_subsets}-subset LDS summary for preferred "
        f"sign={sign} or its inverse under {path}; {details}"
    )


def sign_conversion(preferred: str, stored: str) -> float:
    """Convert an LDS computed with the stored sign to the preferred sign."""
    if preferred not in {"p1", "m1"} or stored not in {"p1", "m1"}:
        raise ValueError(f"unsupported prediction signs: {preferred}, {stored}")
    return 1.0 if preferred == stored else -1.0


def prediction_sign(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("pred_kept_sign_"):
            return parent.name.removeprefix("pred_kept_sign_")
    raise ValueError(f"prediction sign absent from {path}")


def resolve_query_lds_root(
    *, eval_root: Path, prompt: str, seed: int, config: DatasetConfig
) -> Path:
    """Resolve current or legacy-truncated prompt directories without guessing."""
    full_tag = prompt_tag(prompt, config)
    exact = eval_root / f"query_{full_tag}" / f"initial_seed_{seed}" / "lds"
    if exact.is_dir():
        return exact
    matches = []
    for query_dir in eval_root.glob("query_*"):
        stored_tag = query_dir.name.removeprefix("query_")
        if not (full_tag.startswith(stored_tag) or stored_tag.startswith(full_tag)):
            continue
        candidate = query_dir / f"initial_seed_{seed}" / "lds"
        if candidate.is_dir():
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one current/legacy LDS query root for tag={full_tag!r} "
            f"seed={seed}; found {len(matches)}: {matches}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.dataset]
    repo_root = args.repo_root.expanduser().resolve()
    dataset_root = repo_root / config.relative_root
    query_file = (args.query_file or dataset_root / config.query_file).expanduser().resolve()
    result_root = (
        args.result_root or dataset_root / "result" / config.experiment
    ).expanduser().resolve()
    methods = method_configs(args.dataset)
    query_ids = parse_query_ids(args.query_ids)
    selected_targets = tuple(
        value for value in args.targets.replace(",", " ").split() if value
    )
    if not selected_targets:
        raise ValueError("--targets must not be empty")
    invalid_targets = sorted(set(selected_targets) - set(TARGETS))
    if invalid_targets:
        raise ValueError(f"unsupported targets: {invalid_targets}")
    records = json.loads(query_file.read_text())["queries"]
    if min(query_ids) < 0 or max(query_ids) >= len(records):
        raise ValueError(f"query IDs exceed manifest range 0..{len(records) - 1}")
    output = args.output or result_root / "analysis" / "pint_lds_q0_99_all.txt"
    eval_root = result_root / "eval" / "prompted_solo"

    rows: list[dict[str, object]] = []
    selected_transforms = {
        "both": ("RAW-GRADIENT", "ADAMW"),
        "raw-gradient": ("RAW-GRADIENT",),
        "adamw": ("ADAMW",),
    }[args.transforms]
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        query_root = resolve_query_lds_root(
            eval_root=eval_root,
            prompt=prompt,
            seed=seed,
            config=config,
        )
        for method in methods:
            for transform, namespace, sign, multiplier in (
                (
                    "RAW-GRADIENT",
                    method.raw_namespace,
                    method.raw_sign,
                    method.raw_multiplier,
                ),
                (
                    "ADAMW",
                    method.adamw_namespace,
                    method.adamw_sign,
                    method.adamw_multiplier,
                ),
            ):
                if transform not in selected_transforms:
                    continue
                for target in selected_targets:
                    payload, summary_path = load_summary(
                        query_root / f"{namespace}_{method.variant}" / target,
                        args.expected_subsets,
                        sign,
                    )
                    stored = float(payload["lds_percent"])
                    stored_sign = prediction_sign(summary_path)
                    effective_multiplier = multiplier * sign_conversion(
                        sign, stored_sign
                    )
                    rows.append(
                        {
                            "query_id": query_id,
                            "initial_seed": seed,
                            "method": method.label,
                            "transform": transform,
                            "variant": method.variant,
                            "target": target,
                            "stored_lds": stored,
                            "reported_lds": effective_multiplier * stored,
                            "report_multiplier": effective_multiplier,
                            "num_subsets": int(payload["num_models"]),
                            "preferred_sign": sign,
                            "sign": stored_sign,
                            "subset_group": summary_path.parent.name,
                            "summary_path": str(summary_path),
                        }
                    )

    expected = (
        len(query_ids)
        * len(methods)
        * len(selected_transforms)
        * len(selected_targets)
    )
    if len(rows) != expected:
        raise RuntimeError(f"internal coverage error: {len(rows)}/{expected}")

    lookup = {
        (
            int(row["query_id"]),
            str(row["target"]),
            str(row["transform"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    lines = [
        f"{config.title} FIVE-METHOD LDS — RAW-GRADIENT AND ADAMW",
        "=" * 132,
        f"queries={len(query_ids)} range={min(query_ids)}..{max(query_ids)}",
        f"LDS subsets per query/target/configuration={args.expected_subsets}",
        f"expected records={expected} loaded records={len(rows)}",
        "reported LDS applies the method/transform direction multiplier shown in provenance.",
        "Each cell is reported_lds_percent [number_of_subsets].",
        "",
    ]
    columns = tuple(
        (f"{transform} / {method.label}", transform, method.label)
        for transform in selected_transforms
        for method in methods
    )
    for target in selected_targets:
        lines.extend([target.upper(), "-" * 132])
        lines.append(
            f"{'QUERY':<8}" + "".join(f"{label:>28}" for label, _, _ in columns)
        )
        lines.append("-" * 132)
        for query_id in query_ids:
            cells = []
            for _, transform, method in columns:
                row = lookup[(query_id, target, transform, method)]
                cells.append(
                    f"{float(row['reported_lds']):+10.3f}% [{int(row['num_subsets']):3d}]"
                )
            lines.append(f"Q{query_id:<7d}" + "".join(f"{cell:>28}" for cell in cells))
        lines.append("-" * 132)
        for statistic_name, function in (
            ("MEAN", statistics.fmean),
            ("MEDIAN", statistics.median),
            ("STD", statistics.pstdev),
        ):
            values = []
            for _, transform, method in columns:
                series = [
                    float(lookup[(qid, target, transform, method)]["reported_lds"])
                    for qid in query_ids
                ]
                values.append(f"{function(series):+10.3f}%")
            lines.append(f"{statistic_name:<8}" + "".join(f"{value:>28}" for value in values))
        lines.append("")

    lines.extend(["DETAILED PROVENANCE", "=" * 132])
    lines.append(
        "query_id\tseed\ttransform\tmethod\tvariant\ttarget\tstored_lds_percent\t"
        "report_multiplier\treported_lds_percent\tnum_subsets\tpreferred_sign\t"
        "stored_prediction_sign\t"
        "subset_group\tsummary_path"
    )
    for row in rows:
        lines.append(
            "\t".join(
                (
                    str(row["query_id"]),
                    str(row["initial_seed"]),
                    str(row["transform"]),
                    str(row["method"]),
                    str(row["variant"]),
                    str(row["target"]),
                    f"{float(row['stored_lds']):+.9f}",
                    f"{float(row['report_multiplier']):+.0f}",
                    f"{float(row['reported_lds']):+.9f}",
                    str(row["num_subsets"]),
                    str(row["preferred_sign"]),
                    str(row["sign"]),
                    str(row["subset_group"]),
                    str(row["summary_path"]),
                )
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"SAVED: {output}")
    print(f"RECORDS: {len(rows)}")
    print(f"SUBSETS PER RECORD: {args.expected_subsets}")


if __name__ == "__main__":
    main()
