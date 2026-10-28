#!/usr/bin/env python3
"""Export per-query PINT LDS results for CelebA or Shapes3D to one TXT."""

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
)
VARIANTS = (("RAW", "raw"), ("BOTH-L2", "query_train_l2"))


@dataclass(frozen=True)
class DatasetConfig:
    title: str
    relative_root: str
    experiment: str
    query_file: str
    raw_namespace: str
    adamw_namespace: str
    prompt_tag_limit: int
    empty_tag: str


CONFIGS = {
    "celeba": DatasetConfig(
        title="CELEBA",
        relative_root="diffusion_jax_refined/celeba",
        experiment="attr40_experiment1",
        query_file="queries_attr40_val25_x4.json",
        raw_namespace=(
            "traj_tracin_celeba_endpoint_pollute_raw_gradient_train1x50_"
            "query1x50_timestamp_sum_squared_shape3d_semantics"
        ),
        adamw_namespace=(
            "traj_tracin_celeba_endpoint_pollute_adamw_full_train1x50_"
            "query1x50_timestamp_sum_squared_shape3d_semantics"
        ),
        prompt_tag_limit=150,
        empty_tag="empty_prompt",
    ),
    "shape3d": DatasetConfig(
        title="SHAPES3D",
        relative_root="diffusion_jax_refined/3dshapes",
        experiment="experiment1",
        query_file="queries_in_distribution_plus_zero_seed_100_219.json",
        raw_namespace=(
            "traj_tracin_recreate_raw_polluted_endpoint_delta_l2normalized_"
            "timestamp_sum_squared_q0_99"
        ),
        adamw_namespace=(
            "traj_tracin_recreate_adamw_full_polluted_endpoint_"
            "delta_l2normalized_timestamp_sum_squared_q0_99"
        ),
        prompt_tag_limit=160,
        empty_tag="empty",
    ),
}


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
    parser.add_argument("--raw-namespace")
    parser.add_argument("--adamw-namespace")
    parser.add_argument("--expected-subsets", type=int, default=192)
    parser.add_argument("--query-ids", default="0-99")
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


def load_summary(path: Path, expected_subsets: int) -> tuple[dict[str, object], Path]:
    candidates = sorted(path.glob("pred_kept_sign_*/*/lds_summary.json"))
    decoded: list[tuple[dict[str, object], Path]] = []
    for candidate in candidates:
        payload = json.loads(candidate.read_text())
        if int(payload.get("num_models", -1)) == expected_subsets:
            decoded.append((payload, candidate))
    if len(decoded) != 1:
        raise RuntimeError(
            f"expected one {expected_subsets}-subset LDS summary under {path}; "
            f"found {len(decoded)} of {len(candidates)} candidates: {candidates}"
        )
    return decoded[0]


def prediction_sign(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("pred_kept_sign_"):
            return parent.name.removeprefix("pred_kept_sign_")
    raise ValueError(f"prediction sign absent from {path}")


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.dataset]
    repo_root = args.repo_root.expanduser().resolve()
    dataset_root = repo_root / config.relative_root
    query_file = (args.query_file or dataset_root / config.query_file).expanduser().resolve()
    result_root = (
        args.result_root or dataset_root / "result" / config.experiment
    ).expanduser().resolve()
    raw_namespace = args.raw_namespace or config.raw_namespace
    adamw_namespace = args.adamw_namespace or config.adamw_namespace
    query_ids = parse_query_ids(args.query_ids)
    records = json.loads(query_file.read_text())["queries"]
    if min(query_ids) < 0 or max(query_ids) >= len(records):
        raise ValueError(f"query IDs exceed manifest range 0..{len(records) - 1}")
    output = args.output or result_root / "analysis" / "pint_lds_q0_99_all.txt"

    rows: list[dict[str, object]] = []
    transforms = (("RAW-GRADIENT", raw_namespace), ("ADAMW", adamw_namespace))
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        query_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag(prompt, config)}"
            / f"initial_seed_{seed}"
            / "lds"
        )
        for transform, namespace in transforms:
            for variant_label, variant in VARIANTS:
                for target in TARGETS:
                    payload, summary_path = load_summary(
                        query_root / f"{namespace}_{variant}" / target,
                        args.expected_subsets,
                    )
                    stored = float(payload["lds_percent"])
                    # PINT scores encode a loss-reduction direction, so negate the
                    # cached correlation to report a common larger-is-better LDS.
                    rows.append(
                        {
                            "query_id": query_id,
                            "initial_seed": seed,
                            "transform": transform,
                            "variant": variant_label,
                            "target": target,
                            "stored_lds": stored,
                            "reported_lds": -stored,
                            "num_subsets": int(payload["num_models"]),
                            "sign": prediction_sign(summary_path),
                            "subset_group": summary_path.parent.name,
                            "summary_path": str(summary_path),
                        }
                    )

    expected = len(query_ids) * len(transforms) * len(VARIANTS) * len(TARGETS)
    if len(rows) != expected:
        raise RuntimeError(f"internal coverage error: {len(rows)}/{expected}")

    lookup = {
        (int(row["query_id"]), str(row["target"]), str(row["transform"]), str(row["variant"])): row
        for row in rows
    }
    lines = [
        f"{config.title} PINT LDS — RAW/BOTH-L2 × RAW-GRADIENT/ADAMW",
        "=" * 132,
        f"queries={len(query_ids)} range={min(query_ids)}..{max(query_ids)}",
        f"LDS subsets per query/target/configuration={args.expected_subsets}",
        f"expected records={expected} loaded records={len(rows)}",
        "reported LDS = -1 × stored LDS, giving the common larger-is-better PINT orientation.",
        "Each cell is reported_lds_percent [number_of_subsets].",
        "",
    ]
    columns = (
        ("RAW-GRAD / RAW", "RAW-GRADIENT", "RAW"),
        ("RAW-GRAD / BOTH-L2", "RAW-GRADIENT", "BOTH-L2"),
        ("ADAMW / RAW", "ADAMW", "RAW"),
        ("ADAMW / BOTH-L2", "ADAMW", "BOTH-L2"),
    )
    for target in TARGETS:
        lines.extend([target.upper(), "-" * 132])
        lines.append(
            f"{'QUERY':<8}" + "".join(f"{label:>28}" for label, _, _ in columns)
        )
        lines.append("-" * 132)
        for query_id in query_ids:
            cells = []
            for _, transform, variant in columns:
                row = lookup[(query_id, target, transform, variant)]
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
            for _, transform, variant in columns:
                series = [
                    float(lookup[(qid, target, transform, variant)]["reported_lds"])
                    for qid in query_ids
                ]
                values.append(f"{function(series):+10.3f}%")
            lines.append(f"{statistic_name:<8}" + "".join(f"{value:>28}" for value in values))
        lines.append("")

    lines.extend(["DETAILED PROVENANCE", "=" * 132])
    lines.append(
        "query_id\tseed\ttransform\tvariant\ttarget\tstored_lds_percent\t"
        "reported_lds_percent\tnum_subsets\tprediction_sign\tsubset_group\tsummary_path"
    )
    for row in rows:
        lines.append(
            "\t".join(
                (
                    str(row["query_id"]),
                    str(row["initial_seed"]),
                    str(row["transform"]),
                    str(row["variant"]),
                    str(row["target"]),
                    f"{float(row['stored_lds']):+.9f}",
                    f"{float(row['reported_lds']):+.9f}",
                    str(row["num_subsets"]),
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
