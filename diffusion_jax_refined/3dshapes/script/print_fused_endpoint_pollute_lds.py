from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import fmean
import sys


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


REDUCTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)
GROUP = (
    "m_64_k_2500_subset_seed_0__"
    "m_64_k_2500_subset_seed_1__"
    "m_64_k_2500_subset_seed_2"
)


def load_percent(item: tuple[tuple[str, str, int, int], Path]):
    key, path = item
    if not path.is_file():
        raise FileNotFoundError(path)
    return key, float(json.loads(path.read_text())["lds_percent"])


def parse_choices(text: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    invalid = [value for value in values if value not in allowed]
    if not values or invalid:
        raise ValueError(f"invalid {label}: {invalid or text!r}; allowed={allowed}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="p1")
    parser.add_argument("--reductions", default=",".join(REDUCTIONS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--negate-values",
        action="store_true",
        help="Negate stored LDS values, e.g. convert stored P1 Spearman LDS to M1.",
    )
    args = parser.parse_args()
    reductions = parse_choices(args.reductions, REDUCTIONS, "reductions")
    variants = parse_choices(args.variants, VARIANTS, "variants")

    queries = json.loads(args.query_file.read_text())["queries"]
    eval_root = SHAPES_ROOT / "result" / args.experiment / "eval" / "prompted_solo"
    items = []
    for reduction in reductions:
        for variant in variants:
            namespace = (
                "traj_tracin_recreate_adamw_full_"
                "polluted_endpoint_delta_l2normalized_"
                f"{reduction}_aligned100x1_q0_99_{variant}"
            )
            for qid, query in enumerate(queries[:100]):
                seed = int(query["initial_seed"])
                root = (
                    eval_root
                    / f"query_{_prompt_tag(query['prompt'])}"
                    / f"initial_seed_{seed}"
                    / "lds"
                    / namespace
                )
                for target_index, target in enumerate(TARGETS):
                    path = (
                        root
                        / target
                        / f"pred_kept_sign_{args.prediction_sign}"
                        / GROUP
                        / "lds_summary.json"
                    )
                    items.append(((reduction, variant, qid, target_index), path))

    values = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for key, value in executor.map(load_percent, items):
            values[key] = -value if args.negate_values else value

    output_sign = (
        "M1 (CONVERTED FROM STORED P1)"
        if args.negate_values and args.prediction_sign == "p1"
        else "P1 (CONVERTED FROM STORED M1)"
        if args.negate_values and args.prediction_sign == "m1"
        else args.prediction_sign.upper()
    )
    for reduction in reductions:
        for variant in variants:
            print()
            print(
                "ADAMW FULL ENDPOINT-POLLUTE 100x1"
                f" — {reduction.upper()} — {variant.upper()}"
                f" — {output_sign}"
            )
            print(
                f"{'QUERY':7s}{'ENDPOINT':>13s}{'TRAJ-CF':>13s}"
                f"{'SIMPLE':>13s}{'NOISE':>13s}"
            )
            print("-" * 59)
            for qid in range(100):
                row = [values[(reduction, variant, qid, i)] for i in range(4)]
                print(f"Q{qid:<6d}" + "".join(f"{value:+12.3f}%" for value in row))
            means = [
                fmean(values[(reduction, variant, qid, i)] for qid in range(100))
                for i in range(4)
            ]
            print(f"{'MEAN':7s}" + "".join(f"{value:+12.3f}%" for value in means))


if __name__ == "__main__":
    main()
