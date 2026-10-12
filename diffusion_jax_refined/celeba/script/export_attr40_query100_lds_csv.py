#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

CELEBA_ROOT = Path(__file__).resolve().parents[1]
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))

from dataset_config_attr40 import _prompt_path_tag


ALIASES = {
    "endpoint_counterfactual": "endpoint_contarfactual",
    "traj_counterfactual": "traj_contarfactual",
}


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["query_id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CelebA query100 LDS true-f caches to CSV.")
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--subset-seeds", default="0,1,2")
    parser.add_argument(
        "--target-functions",
        default="endpoint_counterfactual,traj_counterfactual,simple_loss,noise_trajectory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.query_file.read_text())["queries"]
    subset_seeds = parse_ints(args.subset_seeds)
    targets = [
        ALIASES.get(value, value)
        for value in args.target_functions.replace(",", " ").split()
        if value
    ]
    rows = []
    missing = []
    for query in records:
        query_root = (
            args.result_root
            / "eval"
            / "prompted_solo"
            / f"query_{_prompt_path_tag(query['prompt'])}"
            / f"initial_seed_{int(query['initial_seed'])}"
            / "lds_target_cache"
            / "ddim_eta0"
        )
        for target in targets:
            matches = sorted(query_root.glob(f"**/{target}/target_*.json"))
            expected = 64 * len(subset_seeds)
            if len(matches) != expected:
                missing.append(
                    f"query={query['query_id']} target={target}: {len(matches)}/{expected}"
                )
                continue
            for path in matches:
                global_id = int(path.stem.rsplit("_", 1)[1])
                payload = json.loads(path.read_text())
                rows.append(
                    {
                        "query_id": int(query["query_id"]),
                        "prompt_id": int(query["prompt_id"]),
                        "source_filename": query["source_filename"],
                        "initial_seed": int(query["initial_seed"]),
                        "prompt": query["prompt"],
                        "target_function": target,
                        "global_subset_id": global_id,
                        "subset_seed": subset_seeds[global_id // 64],
                        "subset_id": global_id % 64,
                        "true_f": float(payload["true_f"]),
                        "checkpoint": payload.get("checkpoint", ""),
                        "subset_dir": payload.get("subset_dir", ""),
                        "trajectory_sampler": payload.get("trajectory_sampler", ""),
                    }
                )
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"LDS cache is incomplete ({len(missing)} groups):\n{preview}")
    rows.sort(
        key=lambda row: (
            int(row["query_id"]), str(row["target_function"]), int(row["global_subset_id"])
        )
    )
    write_rows(args.output_dir / "celeba_attr40_query100_lds_true_f_long.csv", rows)
    for target in targets:
        write_rows(
            args.output_dir / f"celeba_attr40_query100_{target}.csv",
            [row for row in rows if row["target_function"] == target],
        )
    manifest_rows = []
    for query in records:
        manifest_rows.append(
            {
                "query_id": query["query_id"],
                "prompt_id": query["prompt_id"],
                "source_filename": query["source_filename"],
                "initial_seed": query["initial_seed"],
                "prompt": query["prompt"],
            }
        )
    write_rows(args.output_dir / "celeba_attr40_query100_manifest.csv", manifest_rows)
    print(f"saved {len(rows)} LDS rows under {args.output_dir}")


if __name__ == "__main__":
    main()
