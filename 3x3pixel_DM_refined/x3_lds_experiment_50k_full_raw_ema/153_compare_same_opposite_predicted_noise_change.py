"""Compare predicted-noise change directions on +epsilon and -epsilon inputs."""

import json

import numpy as np

from null_gradient_cross_direction_config import *


SOURCES = {
    "actual_null_to_next": (
        "same_actual_delta",
        "opposite_actual_delta",
    ),
    "single_point_sgd_jvp": (
        "same_single_point_sgd_jvp",
        "opposite_single_point_sgd_jvp",
    ),
    "checkpoint_parameter_delta_jvp": (
        "same_checkpoint_delta_jvp",
        "opposite_checkpoint_delta_jvp",
    ),
}


def cosine(left, right, axis=None):
    numerator = np.sum(left * right, axis=axis)
    denominator = np.linalg.norm(left, axis=axis) * np.linalg.norm(
        right, axis=axis
    )
    return numerator / np.maximum(denominator, NGCD_EPS)


def compare(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(T, -1)
    right = np.asarray(right, dtype=np.float64).reshape(T, -1)
    global_cosine = float(cosine(left.reshape(-1), right.reshape(-1)))
    per_timestamp = cosine(left, right, axis=1)
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    best_scalar = float(
        np.sum(left * right) / max(np.sum(left * left), NGCD_EPS)
    )
    residual = right - best_scalar * left
    return {
        "global_cosine": global_cosine,
        "per_timestamp_cosine_mean": float(np.mean(per_timestamp)),
        "per_timestamp_cosine_std": float(np.std(per_timestamp)),
        "per_timestamp_cosine_median": float(np.median(per_timestamp)),
        "per_timestamp_positive_fraction": float(np.mean(per_timestamp > 0)),
        "per_timestamp_negative_fraction": float(np.mean(per_timestamp < 0)),
        "same_norm": float(left_norm),
        "opposite_norm": float(right_norm),
        "opposite_over_same_norm": float(
            right_norm / max(left_norm, NGCD_EPS)
        ),
        "best_scalar_opposite_from_same": best_scalar,
        "best_scaled_relative_residual": float(
            np.linalg.norm(residual) / max(right_norm, NGCD_EPS)
        ),
        "per_timestamp_cosines": per_timestamp.tolist(),
    }


def aggregate(per_datapoint, source):
    keys = (
        "global_cosine",
        "per_timestamp_cosine_mean",
        "per_timestamp_positive_fraction",
        "opposite_over_same_norm",
        "best_scaled_relative_residual",
    )
    return {
        key: {
            "mean": float(
                np.mean([entry[source][key] for entry in per_datapoint])
            ),
            "std": float(
                np.std([entry[source][key] for entry in per_datapoint])
            ),
            "per_datapoint": [
                float(entry[source][key]) for entry in per_datapoint
            ],
        }
        for key in keys
    }


def main():
    per_datapoint = []
    for datapoint_index in nsdl_datapoint_indices():
        path = (
            NGCD_POINT_DIR
            / f"i{datapoint_index:05d}"
            / "predicted_noise_direction_arrays.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run 152_launch_null_gradient_cross_direction_4gpu.py first"
            )
        arrays = np.load(path, allow_pickle=False)
        entry = {"datapoint_index": int(datapoint_index)}
        for source, (same_key, opposite_key) in SOURCES.items():
            entry[source] = compare(arrays[same_key], arrays[opposite_key])
        per_datapoint.append(entry)

    output = {
        "question": (
            "Are predicted-noise changes on same- and opposite-noise inputs "
            "pointing in the same output-space direction?"
        ),
        "interpretation": (
            "cosine near +1 means same direction; near 0 means unrelated; "
            "near -1 means opposite direction"
        ),
        "null_epoch": NGCD_NULL_EPOCH,
        "next_epoch": NGCD_NEXT_EPOCH,
        "datapoint_indices": list(nsdl_datapoint_indices()),
        "sources": {
            source: aggregate(per_datapoint, source) for source in SOURCES
        },
        "per_datapoint_results": per_datapoint,
    }
    output_path = NGCD_ROOT / "same_opposite_change_direction_consistency.json"
    NGCD_ROOT.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)

    for source in SOURCES:
        result = output["sources"][source]
        cosine_result = result["global_cosine"]
        positive = result["per_timestamp_positive_fraction"]
        ratio = result["opposite_over_same_norm"]
        print(
            f"{source:34s} same-vs-opposite cosine="
            f"{cosine_result['mean']:+.6f} ± {cosine_result['std']:.6f} | "
            f"positive timestamps={positive['mean']:.3f} | "
            f"norm ratio={ratio['mean']:.4f}",
            flush=True,
        )
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
