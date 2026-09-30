"""Compare predicted-noise changes on same and opposite noising directions."""

import argparse
import json
import os

import numpy as np

from null_gradient_cross_direction_config import (
    NGCD_EPS,
    ngcd_odd_even_output_paths,
    nsdl_datapoint_indices,
)


SOURCES = (
    "actual_delta",
    "plus_loss_sgd_jvp",
    "minus_loss_sgd_jvp",
    "even_loss_sgd_jvp",
    "odd_loss_sgd_jvp",
    "checkpoint_parameter_delta_jvp",
)


def comparison(same, opposite):
    same64 = same.astype(np.float64)
    opposite64 = opposite.astype(np.float64)
    same_flat = same64.reshape(-1)
    opposite_flat = opposite64.reshape(-1)
    cosine = float(
        np.dot(same_flat, opposite_flat)
        / max(np.linalg.norm(same_flat) * np.linalg.norm(opposite_flat), NGCD_EPS)
    )
    same_t = same64.reshape(len(same64), -1)
    opposite_t = opposite64.reshape(len(opposite64), -1)
    denominators = np.maximum(
        np.linalg.norm(same_t, axis=1) * np.linalg.norm(opposite_t, axis=1),
        NGCD_EPS,
    )
    per_timestamp = np.sum(same_t * opposite_t, axis=1) / denominators
    return {
        "global_cosine": cosine,
        "per_timestamp_cosine_mean": float(per_timestamp.mean()),
        "per_timestamp_cosine_std": float(per_timestamp.std()),
        "per_timestamp_positive_fraction": float(np.mean(per_timestamp > 0.0)),
        "opposite_over_same_norm": float(
            np.linalg.norm(opposite_flat) / max(np.linalg.norm(same_flat), NGCD_EPS)
        ),
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-prompt", choices=("original", "random"), default="random"
    )
    args = parser.parse_args()
    root, point_dir, _, _ = ngcd_odd_even_output_paths(args.evaluation_prompt)
    per_datapoint = []
    for datapoint_index in nsdl_datapoint_indices():
        path = point_dir / f"i{datapoint_index:05d}" / "odd_even_direction_arrays.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as arrays:
            comparisons = {
                source: comparison(arrays[f"same_{source}"], arrays[f"opposite_{source}"])
                for source in SOURCES
            }
        per_datapoint.append(
            {"datapoint_index": int(datapoint_index), "sources": comparisons}
        )

    result = {
        "evaluation_prompt_mode": args.evaluation_prompt,
        "datapoint_indices": [entry["datapoint_index"] for entry in per_datapoint],
        "sources": {},
    }
    for source in SOURCES:
        result["sources"][source] = {}
        metric_names = per_datapoint[0]["sources"][source].keys()
        for metric in metric_names:
            values = np.asarray(
                [entry["sources"][source][metric] for entry in per_datapoint],
                dtype=np.float64,
            )
            result["sources"][source][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "per_datapoint": values.tolist(),
            }
    output_path = root / "odd_even_same_opposite_consistency.json"
    atomic_json(output_path, result)
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
