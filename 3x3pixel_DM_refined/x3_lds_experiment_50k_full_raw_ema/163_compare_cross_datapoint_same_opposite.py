"""Compare target responses on positive and negative target noise directions."""

import argparse
import importlib
import json
import os

import numpy as np

from null_gradient_cross_direction_config import *


odd_even_analysis = importlib.import_module(
    "160_compare_null_gradient_odd_even_same_opposite"
)
SOURCES = (
    "actual_delta",
    "plus_loss_sgd_jvp",
    "even_loss_sgd_jvp",
    "checkpoint_parameter_delta_jvp",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-noise-mode",
        choices=("shared", "independent"),
        default="shared",
    )
    args = parser.parse_args()
    root, pair_dir, _, _ = ngcd_cross_datapoint_output_paths(
        args.target_noise_mode
    )
    entries = []
    for source_index in nsdl_datapoint_indices():
        target_index = ngcd_cross_datapoint_target_index(source_index)
        path = (
            pair_dir
            / f"source_{source_index:05d}_target_{target_index:05d}"
            / "cross_datapoint_direction_arrays.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as arrays:
            comparisons = {
                source: odd_even_analysis.comparison(
                    arrays[f"same_{source}"], arrays[f"opposite_{source}"]
                )
                for source in SOURCES
            }
        entries.append(
            {
                "source_datapoint_index": int(source_index),
                "target_datapoint_index": int(target_index),
                "sources": comparisons,
            }
        )

    result = {
        "target_noise_mode": args.target_noise_mode,
        "pairs": [
            {
                "source": entry["source_datapoint_index"],
                "target": entry["target_datapoint_index"],
            }
            for entry in entries
        ],
        "sources": {},
    }
    for source in SOURCES:
        result["sources"][source] = {}
        for metric in entries[0]["sources"][source]:
            values = np.asarray(
                [entry["sources"][source][metric] for entry in entries],
                dtype=np.float64,
            )
            result["sources"][source][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "per_pair": values.tolist(),
            }
    output_path = root / "cross_datapoint_same_opposite_consistency.json"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    os.replace(temporary, output_path)
    for source in SOURCES:
        cosine = result["sources"][source]["global_cosine"]
        positive = result["sources"][source]["per_timestamp_positive_fraction"]
        ratio = result["sources"][source]["opposite_over_same_norm"]
        print(
            f"{source:34s} same-vs-opposite cosine={cosine['mean']:+.6f} ± {cosine['std']:.6f} "
            f"| positive timestamps={positive['mean']:.3f} | norm ratio={ratio['mean']:.4f}",
            flush=True,
        )
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
