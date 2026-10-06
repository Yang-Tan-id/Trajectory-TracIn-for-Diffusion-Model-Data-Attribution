"""Whiten the timestamp dimension of cached endpoint20 responses."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from endpoint20_meanloss_pairing_config import *


DEFAULT_LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def lambda_label(value):
    if value == 0:
        return "0"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def evaluate(score, membership, observed):
    prediction = membership @ score.T
    return {
        metric: np.asarray(
            [
                spearmanr(-prediction[:, q], target[q]).statistic
                for q in range(score.shape[0])
            ],
            dtype=np.float64,
        )
        for metric, target in observed.items()
    }


def paired_test(candidate, baseline):
    difference = candidate - baseline
    finite = difference[np.isfinite(difference)]
    try:
        pvalue = float(wilcoxon(finite, alternative="two-sided").pvalue)
    except ValueError:
        pvalue = 1.0
    return {
        "mean_difference": float(np.nanmean(difference)),
        "wilcoxon_two_sided_p": pvalue,
        "wins": int(np.count_nonzero(difference > 0)),
        "query_count": int(len(finite)),
        "difference_per_query": difference.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-shard-count", type=int, default=4)
    parser.add_argument(
        "--mode", choices=E20_MODES, default="per_timestamp_aligned"
    )
    parser.add_argument(
        "--lambdas",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated shrinkage strengths relative to mean eigenvalue.",
    )
    args = parser.parse_args()
    lambdas = tuple(float(value) for value in args.lambdas.split(","))

    response = np.zeros(
        (len(E20_QUERY_IDS), len(E20_TIMESTEPS), N_TRAIN),
        dtype=np.float64,
    )
    covered = []
    key = f"{args.mode}__timestamp_response"
    for shard_index in range(args.checkpoint_shard_count):
        root = e20_shard_root(shard_index, args.checkpoint_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        covered.extend(int(value) for value in info["checkpoint_pairs"])
        with np.load(root / "partial_scores.npz") as partial:
            response += partial[key].astype(np.float64)
    if sorted(covered) != sorted(NPA_CHECKPOINT_PAIRS):
        raise ValueError(f"checkpoint coverage mismatch: {sorted(covered)}")

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(E20_QUERY_IDS)
        ]
        for metric in LDS_METRICS
    }

    # Uniform timestamp energy is the current timestamp-sum-squared score.
    scores = {"uniform": np.square(response).mean(axis=1)}
    spectra = []
    whitened = {
        value: np.empty((len(E20_QUERY_IDS), N_TRAIN), dtype=np.float64)
        for value in lambdas
    }
    for query_position in range(len(E20_QUERY_IDS)):
        matrix = response[query_position]  # [timestamp, training datapoint]
        # Use an uncentered second moment because zero response has a physical
        # meaning; centering would redefine a zero-effect datapoint.
        second_moment = matrix @ matrix.T / matrix.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        scale = max(float(np.mean(eigenvalues)), np.finfo(np.float64).tiny)
        floor = scale * 1e-12
        coordinates = eigenvectors.T @ matrix
        spectra.append(
            {
                "query_id": int(E20_QUERY_IDS[query_position]),
                "eigenvalues": eigenvalues.tolist(),
                "condition_with_floor": float(
                    eigenvalues[-1] / max(eigenvalues[0], floor)
                ),
                "effective_rank": float(
                    np.square(eigenvalues.sum())
                    / max(np.square(eigenvalues).sum(), floor)
                ),
            }
        )
        for value in lambdas:
            denominator = eigenvalues + value * scale
            denominator = np.maximum(denominator, floor)
            whitened[value][query_position] = (
                np.square(coordinates) / denominator[:, None]
            ).sum(axis=0)

    for value, score in whitened.items():
        scores[f"whiten_lambda_{lambda_label(value)}"] = score

    lds = {
        name: evaluate(score, membership, observed)
        for name, score in scores.items()
    }
    baseline = lds["uniform"]
    result = {
        "mode": args.mode,
        "query_ids": list(E20_QUERY_IDS),
        "timesteps": list(E20_TIMESTEPS),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "whitening": "uncentered timestamp second moment",
        "regularization": "lambda times per-query mean eigenvalue",
        "spectra": spectra,
        "results": {},
    }
    lines = [
        "ENDPOINT20 TIMESTAMP SECOND-MOMENT WHITENING",
        f"mode={args.mode}; all 20 timestamps; sign=-1; q00-q09",
        "No timestamp selection; all methods use identical cached responses.",
        "",
    ]
    for name, target_values in lds.items():
        lines.append(f"[{name}]")
        lines.append(
            "target                              mean       std      "
            "delta-vs-uniform       p       wins"
        )
        result["results"][name] = {}
        for metric in LDS_METRICS:
            values = target_values[metric]
            test = paired_test(values, baseline[metric])
            result["results"][name][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "per_query": values.tolist(),
                "vs_uniform": test,
            }
            lines.append(
                f"{metric:34s} {np.nanmean(values):+.6f} "
                f"{np.nanstd(values):.6f} {test['mean_difference']:+.6f} "
                f"{test['wilcoxon_two_sided_p']:.6g} {test['wins']:02d}/10"
            )
        lines.append("")

    effective_ranks = np.asarray(
        [entry["effective_rank"] for entry in spectra], dtype=np.float64
    )
    lines.append(
        "timestamp effective rank: "
        f"mean={effective_ranks.mean():.4f} std={effective_ranks.std():.4f} "
        f"min={effective_ranks.min():.4f} max={effective_ranks.max():.4f}"
    )

    stem = f"tracin_das_endpoint20_{args.mode}_timestamp_whitening_q00_q09"
    json_path = LDS_DIR / f"{stem}.json"
    text_path = LDS_DIR / f"{stem}.txt"
    atomic_json(json_path, result)
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
