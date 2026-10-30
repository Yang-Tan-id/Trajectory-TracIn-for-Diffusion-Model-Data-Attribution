"""Compare Raw-gradient and AdamW cross-query Top-K uniqueness on ArtBench."""

from __future__ import annotations

import argparse
import csv
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArtBenchConfig


TRANSFORMS = ("raw_gradient", "adamw_full")

# method, artifact, variant, Raw-gradient sign, AdamW sign
# The sign converts stored scores to a common larger-is-more-influential
# orientation before Top-K selection.
METHODS = (
    ("PINT / RAW", "pollute", "raw", +1, +1),
    ("PINT / BOTH-L2", "pollute", "query_train_l2", +1, +1),
    ("ENDPOINT TRACIN / RAW", "tracin", "raw", +1, -1),
    (
        "ENDPOINT TRACIN / BOTH-L2",
        "tracin",
        "query_train_l2",
        +1,
        -1,
    ),
    ("RETRAC / BOTH-L2", "retrac", None, +1, -1),
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {
            "path": path,
            "scores": np.asarray(payload["scores"], dtype=np.float64),
            "indices": np.asarray(
                payload["train_metadata_indices"], dtype=np.int64
            ).reshape(-1),
            "transforms": [str(value) for value in payload["transform_names"]],
            "variants": (
                [str(value) for value in payload["variant_names"]]
                if "variant_names" in payload.files
                else None
            ),
        }


def score_bank(
    artifact: dict[str, Any], *, transform: str, variant: str | None
) -> np.ndarray:
    try:
        transform_slot = artifact["transforms"].index(transform)
    except ValueError as error:
        raise ValueError(
            f"{artifact['path']}: missing transform {transform!r}"
        ) from error
    scores = artifact["scores"]
    if variant is None:
        bank = scores[transform_slot]
    else:
        variants = artifact["variants"]
        if variants is None:
            raise ValueError(f"{artifact['path']}: no variant axis")
        try:
            variant_slot = variants.index(variant)
        except ValueError as error:
            raise ValueError(
                f"{artifact['path']}: missing variant {variant!r}"
            ) from error
        bank = scores[transform_slot, variant_slot]
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 2:
        raise ValueError(
            f"{artifact['path']}: expected [query, train], found {bank.shape}"
        )
    if bank.shape[1] != len(artifact["indices"]):
        raise ValueError(f"{artifact['path']}: score/index width mismatch")
    return bank


def select_topk(
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    topk: int,
    ranking_sign: int,
) -> set[int]:
    if ranking_sign not in (-1, +1):
        raise ValueError("ranking_sign must be -1 or +1")
    oriented = ranking_sign * np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(oriented)
    if int(finite.sum()) < topk:
        raise ValueError(f"fewer than {topk} finite scores")
    ranking = np.where(finite, oriented, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:topk]
    return set(map(int, indices[order]))


def describe(values: list[float]) -> tuple[float, float, float, float, float]:
    return (
        statistics.fmean(values),
        statistics.median(values),
        statistics.pstdev(values),
        min(values),
        max(values),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=root / "outputs/attribution/endpoint-q60-t50-full-linear-hooks",
    )
    parser.add_argument(
        "--retrac-path",
        type=Path,
        default=(
            root
            / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz"
        ),
    )
    parser.add_argument(
        "--query-indices", type=parse_ints, default=list(range(60))
    )
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_indices = args.query_indices
    if len(query_indices) < 2 or len(query_indices) != len(set(query_indices)):
        raise ValueError("--query-indices must contain at least two unique IDs")

    artifacts = {
        "pollute": load_artifact(
            args.endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz"
        ),
        "tracin": load_artifact(args.endpoint_root / "endpoint_tracin.npz"),
        "retrac": load_artifact(args.retrac_path),
    }
    universes = {
        name: set(map(int, artifact["indices"]))
        for name, artifact in artifacts.items()
    }
    if len({frozenset(values) for values in universes.values()}) != 1:
        raise ValueError("the three attribution artifacts use different candidates")
    universe_size = len(next(iter(universes.values())))
    if not 0 < args.topk < universe_size:
        raise ValueError(f"--topk must be in [1, {universe_size - 1}]")

    output_dir = args.output_dir
    if output_dir is None:
        project = ArtBenchConfig().index_root.parent.parent
        output_dir = project / "outputs/analysis" / (
            f"raw_gradient_adamw_cross_query_top{args.topk}_overlap_"
            f"q{min(query_indices)}_q{max(query_indices)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selections: dict[tuple[str, str], dict[int, set[int]]] = {}
    for method, artifact_name, variant, raw_sign, adamw_sign in METHODS:
        artifact = artifacts[artifact_name]
        for transform, ranking_sign in zip(
            TRANSFORMS, (raw_sign, adamw_sign), strict=True
        ):
            bank = score_bank(artifact, transform=transform, variant=variant)
            missing = [query for query in query_indices if query >= bank.shape[0]]
            if missing:
                raise IndexError(
                    f"{method} {transform}: missing query IDs {missing}"
                )
            selections[(method, transform)] = {
                query: select_topk(
                    bank[query],
                    artifact["indices"],
                    topk=args.topk,
                    ranking_sign=ranking_sign,
                )
                for query in query_indices
            }
            print(
                f"[loaded] {method} {transform} queries={len(query_indices)} "
                f"direction={'descending' if ranking_sign == 1 else 'ascending'}",
                flush=True,
            )

    pair_rows: list[dict[str, object]] = []
    per_query_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pair_values: dict[tuple[str, str], list[float]] = {}

    for method, *_ in METHODS:
        for transform in TRANSFORMS:
            method_sets = selections[(method, transform)]
            values: list[float] = []
            for left, right in combinations(query_indices, 2):
                overlap_count = len(method_sets[left] & method_sets[right])
                overlap = overlap_count / args.topk
                values.append(overlap)
                pair_rows.append(
                    {
                        "method": method,
                        "transform": transform,
                        "left_query": left,
                        "right_query": right,
                        "overlap_count": overlap_count,
                        "overlap_fraction": overlap,
                        "unique_fraction": 1.0 - overlap,
                        "jaccard": overlap_count / (2 * args.topk - overlap_count),
                    }
                )
            pair_values[(method, transform)] = values
            stats = describe(values)
            union = set().union(*(method_sets[q] for q in query_indices))
            intersection = set.intersection(
                *(method_sets[q] for q in query_indices)
            )
            jaccards = [value / (2.0 - value) for value in values]
            summary_rows.append(
                {
                    "method": method,
                    "transform": transform,
                    "queries": len(query_indices),
                    "query_pairs": len(values),
                    "topk": args.topk,
                    "candidate_count": universe_size,
                    "random_overlap_expectation": args.topk / universe_size,
                    "overlap_mean": stats[0],
                    "overlap_median": stats[1],
                    "overlap_std": stats[2],
                    "overlap_min": stats[3],
                    "overlap_max": stats[4],
                    "unique_fraction_mean": 1.0 - stats[0],
                    "jaccard_mean": statistics.fmean(jaccards),
                    "union_count": len(union),
                    "intersection_all_queries_count": len(intersection),
                }
            )
            for query in query_indices:
                overlaps = [
                    len(method_sets[query] & method_sets[other]) / args.topk
                    for other in query_indices
                    if other != query
                ]
                others_union = set().union(
                    *(method_sets[other] for other in query_indices if other != query)
                )
                per_query_rows.append(
                    {
                        "method": method,
                        "transform": transform,
                        "query_index": query,
                        "mean_overlap_with_other_queries": statistics.fmean(overlaps),
                        "mean_unique_fraction": 1.0 - statistics.fmean(overlaps),
                        "exclusive_to_this_query_count": len(
                            method_sets[query] - others_union
                        ),
                    }
                )

    comparison_rows: list[dict[str, object]] = []
    for method, *_ in METHODS:
        raw = pair_values[(method, "raw_gradient")]
        adamw = pair_values[(method, "adamw_full")]
        deltas = [right - left for left, right in zip(raw, adamw, strict=True)]
        comparison_rows.append(
            {
                "method": method,
                "query_pairs": len(deltas),
                "raw_gradient_overlap_mean": statistics.fmean(raw),
                "adamw_overlap_mean": statistics.fmean(adamw),
                "adamw_minus_raw_overlap": statistics.fmean(deltas),
                "raw_gradient_unique_mean": 1.0 - statistics.fmean(raw),
                "adamw_unique_mean": 1.0 - statistics.fmean(adamw),
                "adamw_minus_raw_unique": -statistics.fmean(deltas),
                "adamw_lower_overlap_pairs": sum(delta < 0.0 for delta in deltas),
                "adamw_higher_overlap_pairs": sum(delta > 0.0 for delta in deltas),
                "tied_pairs": sum(delta == 0.0 for delta in deltas),
            }
        )

    lines = [
        "ARTBENCH RAW-GRADIENT VS ADAMW CROSS-QUERY TOP-K UNIQUENESS",
        f"queries={len(query_indices)} ({min(query_indices)}..{max(query_indices)}) topk={args.topk} candidates={universe_size}",
        f"query pairs per configuration={len(query_indices) * (len(query_indices) - 1) // 2}",
        f"independent-random overlap expectation={args.topk / universe_size:.2%}",
        "lower overlap means more query-specific; uniqueness=1-overlap",
        "",
        (
            f"{'METHOD':30s} {'RAW-OV':>9s} {'ADAMW-OV':>9s} "
            f"{'DELTA-OV':>9s} {'RAW-UNIQ':>10s} {'ADAMW-UNIQ':>11s} "
            f"{'DELTA-UNIQ':>11s} {'ADAMW LOWER':>13s}"
        ),
        "-" * 112,
    ]
    for row in comparison_rows:
        lines.append(
            f"{str(row['method']):30s} "
            f"{float(row['raw_gradient_overlap_mean']):9.2%} "
            f"{float(row['adamw_overlap_mean']):9.2%} "
            f"{float(row['adamw_minus_raw_overlap']):+9.2%} "
            f"{float(row['raw_gradient_unique_mean']):10.2%} "
            f"{float(row['adamw_unique_mean']):11.2%} "
            f"{float(row['adamw_minus_raw_unique']):+11.2%} "
            f"{int(row['adamw_lower_overlap_pairs']):5d}/{int(row['query_pairs']):<7d}"
        )
    report = "\n".join(lines) + "\n"
    print("\n" + report, end="")

    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "raw_gradient_vs_adamw.csv", comparison_rows)
    write_csv(output_dir / "pairwise_cross_query_overlap.csv", pair_rows)
    write_csv(output_dir / "per_query_uniqueness.csv", per_query_rows)
    report_path = output_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
