"""Post-hoc scalar calibration for two-checkpoint AdamW approximations.

For every checkpoint interval and frozen-gradient method, fit the scalar in
parameter space

    alpha = <d, Delta theta> / ||d||^2,

where d is the approximate checkpoint parameter displacement and Delta theta
is the observed displacement between the two saved checkpoints.  Since JVP is
linear in its tangent, previously saved query-response metrics are sufficient
to evaluate alpha * J d without rerunning a model.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_endpoint_adam_diagnostic_config import CEAD_METHODS, cead_root
from exp_config import FAMILIES


FROZEN_METHODS = CEAD_METHODS[1:]
EPS = 1e-30


def parse_indices(text):
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def calibrated_metrics(actual, predicted, cosine, alpha):
    scaled_norm = abs(alpha) * predicted
    signed_dot = alpha * cosine * actual * predicted
    error_sq = np.maximum(
        np.square(actual) + np.square(alpha * predicted) - 2.0 * signed_dot,
        0.0,
    )
    vector_error = np.sqrt(error_sq) / np.maximum(actual, EPS)
    magnitude_error = np.abs(scaled_norm - actual) / np.maximum(actual, EPS)
    calibrated_cosine = np.sign(alpha) * cosine
    if alpha == 0.0:
        calibrated_cosine = np.zeros_like(cosine)
    return {
        "vector_cosine": calibrated_cosine,
        "vector_relative_error": vector_error,
        "magnitude_relative_error": magnitude_error,
        "predicted_l2": scaled_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-indices", default="0,16,32,48")
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--loss-mc", type=int, default=1)
    args = parser.parse_args()

    pair_indices = parse_indices(args.pair_indices)
    root = cead_root(args.loss_mc)
    pooled = {
        method: {key: [] for key in (
            "actual_l2",
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        )}
        for method in FROZEN_METHODS
    }
    output = {
        "family": args.family,
        "loss_mc": args.loss_mc,
        "pair_indices": list(pair_indices),
        "calibration": "parameter-space least-squares scalar per checkpoint pair",
        "pairs": {},
        "methods": {},
    }

    print("parameter-space scalar calibration by checkpoint pair", flush=True)
    print("pair  method                                      alpha       param-cos", flush=True)
    for pair_index in pair_indices:
        array_path = root / args.family / f"pair_{pair_index:02d}.npz"
        metadata_path = root / args.family / f"pair_{pair_index:02d}.json"
        if not array_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"missing pair {pair_index}: expected {array_path} and {metadata_path}"
            )
        with np.load(array_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].astype(np.float64) for name in payload.files}
        with open(metadata_path) as handle:
            metadata = json.load(handle)

        actual = arrays["actual_l2"]
        pair_output = {}
        for method in FROZEN_METHODS:
            agreement = metadata["parameter_agreement"][method]
            predicted_parameter_norm = float(agreement["predicted_norm"])
            actual_parameter_norm = float(agreement["actual_norm"])
            parameter_cosine = float(agreement["cosine"])
            alpha = (
                parameter_cosine * actual_parameter_norm
                / max(predicted_parameter_norm, EPS)
            )

            predicted = arrays[f"{method}_predicted_l2"]
            cosine = arrays[f"{method}_vector_cosine"]
            metrics = calibrated_metrics(actual, predicted, cosine, alpha)
            pair_output[method] = {
                "alpha": alpha,
                "parameter_cosine": parameter_cosine,
                "parameter_predicted_norm": predicted_parameter_norm,
                "parameter_actual_norm": actual_parameter_norm,
                "calibrated_vector_cosine_mean": float(
                    np.nanmean(metrics["vector_cosine"])
                ),
                "calibrated_vector_relative_error_mean": float(
                    np.nanmean(metrics["vector_relative_error"])
                ),
                "calibrated_magnitude_relative_error_mean": float(
                    np.nanmean(metrics["magnitude_relative_error"])
                ),
                "calibrated_magnitude_ratio": float(
                    metrics["predicted_l2"].sum() / max(actual.sum(), EPS)
                ),
            }
            pooled[method]["actual_l2"].append(actual)
            for key, value in metrics.items():
                pooled[method][key].append(value)
            print(
                f"{pair_index:02d}    {method:44s} "
                f"{alpha:+.6e}  {parameter_cosine:+.6f}",
                flush=True,
            )
        output["pairs"][str(pair_index)] = pair_output

    print("\ncalibrated predicted-noise response", flush=True)
    print(
        "method                                      vector-cos  vector-relerr  "
        "mag-relerr  mag-ratio",
        flush=True,
    )
    for method in FROZEN_METHODS:
        values = {
            key: np.concatenate(chunks) for key, chunks in pooled[method].items()
        }
        summary = {
            "vector_cosine_mean": float(np.nanmean(values["vector_cosine"])),
            "vector_relative_error_mean": float(
                np.nanmean(values["vector_relative_error"])
            ),
            "magnitude_relative_error_mean": float(
                np.nanmean(values["magnitude_relative_error"])
            ),
            "magnitude_ratio": float(
                values["predicted_l2"].sum()
                / max(values["actual_l2"].sum(), EPS)
            ),
        }
        output["methods"][method] = summary
        print(
            f"{method:44s} {summary['vector_cosine_mean']:+.6f}  "
            f"{summary['vector_relative_error_mean']:.6f}       "
            f"{summary['magnitude_relative_error_mean']:.6f}   "
            f"{summary['magnitude_ratio']:.6f}",
            flush=True,
        )

    output_path = root / f"scalar_calibration_{args.family}.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
