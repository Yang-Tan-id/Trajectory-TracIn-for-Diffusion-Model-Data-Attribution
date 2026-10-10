#!/usr/bin/env python3
"""Analyze per-probe proper scores after query-specific timestamp flips."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR = Path("/Users/rachelsomething/Downloads/probe24_timestamp_signs_3506389")
OUTPUT_DIR = Path("analysis_outputs/probe24_proper_score_signs")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_payload = np.load(INPUT_DIR / "per_probe_timestamp_scores.npz")
    sign_payload = np.load(INPUT_DIR / "individual_probe_best_binary_vectors.npz")

    query_ids = raw_payload["query_ids"]
    timesteps = raw_payload["timesteps"]
    score_indices = raw_payload["score_indices"]
    np.testing.assert_array_equal(query_ids, sign_payload["query_ids"])
    np.testing.assert_array_equal(timesteps, sign_payload["timesteps"])
    np.testing.assert_array_equal(score_indices, sign_payload["score_indices"])

    # Raw axes: probe, timestamp, query, datapoint.
    raw = np.concatenate([raw_payload["old12"], raw_payload["fresh12"]], axis=0)
    raw = np.transpose(raw, (0, 2, 1, 3))  # probe, query, timestamp, datapoint
    best_signs = sign_payload["best_signs"].astype(np.float32)
    if not np.all(np.isin(best_signs, (-1, 1))):
        raise ValueError("best_signs must contain only -1 and +1")

    oriented_components = raw * best_signs[..., None]
    reconstructed_binary = oriented_components > 0
    binary = sign_payload["best_binary"]
    mismatch = int(np.count_nonzero(reconstructed_binary != binary))
    if mismatch:
        raise ValueError(f"oriented component sign validation failed: {mismatch} mismatches")

    # Proper score requested by the user: first orient each timestamp, then sum all 10.
    proper = oriented_components.sum(axis=2)  # probe, query, datapoint
    ensemble = proper.mean(axis=0)  # query, datapoint; mean and sum have identical sign/rank
    positive_probe_count = (proper > 0).sum(axis=0)  # query, datapoint

    probe_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    datapoint_rows: list[dict[str, object]] = []

    for qslot, query in enumerate(query_ids.tolist()):
        for probe in range(24):
            values = proper[probe, qslot]
            probe_rows.append(
                {
                    "query": query,
                    "global_probe": probe + 1,
                    "bank": "old12" if probe < 12 else "fresh12",
                    "probe_in_bank": probe + 1 if probe < 12 else probe - 11,
                    "positive_count": int(np.count_nonzero(values > 0)),
                    "negative_count": int(np.count_nonzero(values < 0)),
                    "zero_count": int(np.count_nonzero(values == 0)),
                    "positive_fraction": float(np.mean(values > 0)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

        indiv = proper[:, qslot, :]
        ens = ensemble[qslot]
        counts = positive_probe_count[qslot]
        query_rows.append(
            {
                "query": query,
                "individual_score_count": int(indiv.size),
                "individual_positive_fraction": float(np.mean(indiv > 0)),
                "individual_negative_fraction": float(np.mean(indiv < 0)),
                "old12_positive_fraction": float(np.mean(indiv[:12] > 0)),
                "fresh12_positive_fraction": float(np.mean(indiv[12:] > 0)),
                "ensemble_positive_count": int(np.count_nonzero(ens > 0)),
                "ensemble_negative_count": int(np.count_nonzero(ens < 0)),
                "ensemble_zero_count": int(np.count_nonzero(ens == 0)),
                "ensemble_positive_fraction": float(np.mean(ens > 0)),
                "ensemble_mean": float(np.mean(ens)),
                "ensemble_std": float(np.std(ens)),
                "ensemble_median": float(np.median(ens)),
                "mean_positive_probe_count": float(np.mean(counts)),
                "median_positive_probe_count": float(np.median(counts)),
            }
        )

        for islot, score_index in enumerate(score_indices.tolist()):
            datapoint_rows.append(
                {
                    "query": query,
                    "score_index": score_index,
                    "positive_probes": int(counts[islot]),
                    "negative_probes": int(24 - counts[islot]),
                    "ensemble_proper_score": float(ens[islot]),
                    "ensemble_sign": 1 if ens[islot] > 0 else (-1 if ens[islot] < 0 else 0),
                }
            )

    write_csv(
        OUTPUT_DIR / "per_probe_query_proper_score_summary.csv",
        list(probe_rows[0]),
        probe_rows,
    )
    write_csv(
        OUTPUT_DIR / "per_query_proper_score_summary.csv",
        list(query_rows[0]),
        query_rows,
    )
    write_csv(
        OUTPUT_DIR / "per_datapoint_ensemble_proper_scores.csv",
        list(datapoint_rows[0]),
        datapoint_rows,
    )

    # Figure 1: pooled individual-probe proper scores (24 x 5000 per query).
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    for qslot, ax in enumerate(axes.flat):
        values = proper[:, qslot, :].ravel()
        ax.hist(values, bins=80, color="#4c78a8", alpha=0.9)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(
            f"Query {query_ids[qslot]}\npositive={np.mean(values > 0):.1%}"
        )
        ax.set_xlabel("proper score")
        ax.set_ylabel("count")
    fig.suptitle("24 individual probes after best timestamp flips: sum over 10 timestamps")
    fig.savefig(OUTPUT_DIR / "individual_probe_proper_score_distributions.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "individual_probe_proper_score_distributions.svg")
    plt.close(fig)

    # Figure 2: actual 24-probe ensemble proper score over 5000 datapoints.
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    for qslot, ax in enumerate(axes.flat):
        values = ensemble[qslot]
        ax.hist(values, bins=70, color="#f58518", alpha=0.9)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(np.mean(values), color="#b22222", linewidth=1.4, linestyle="--")
        ax.set_title(
            f"Query {query_ids[qslot]}\npositive={np.mean(values > 0):.1%}"
        )
        ax.set_xlabel("24-probe mean proper score")
        ax.set_ylabel("datapoints")
    fig.suptitle("24-probe ensemble after best timestamp flips: 5000 proper scores")
    fig.savefig(OUTPUT_DIR / "ensemble_proper_score_distributions.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "ensemble_proper_score_distributions.svg")
    plt.close(fig)

    # Figure 3: how many of the 24 full proper scores are positive per datapoint.
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    bins = np.arange(-0.5, 25.5, 1)
    for qslot, ax in enumerate(axes.flat):
        values = positive_probe_count[qslot]
        ax.hist(values, bins=bins, color="#54a24b", alpha=0.9)
        ax.axvline(12, color="black", linewidth=1)
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 4))
        ax.set_title(
            f"Query {query_ids[qslot]}\nmean positive probes={np.mean(values):.2f}/24"
        )
        ax.set_xlabel("positive full-probe proper scores")
        ax.set_ylabel("datapoints")
    fig.suptitle("Probe agreement after summing all 10 timestamps")
    fig.savefig(OUTPUT_DIR / "positive_probe_count_after_timestamp_sum.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "positive_probe_count_after_timestamp_sum.svg")
    plt.close(fig)

    print("Q  INDIV+  OLD12+  FRESH12+  ENSEMBLE+  ENSEMBLE POS/NEG  MEAN +PROBES")
    print("-" * 86)
    for row in query_rows:
        print(
            f"{row['query']:>1d}  "
            f"{row['individual_positive_fraction']:>6.1%}  "
            f"{row['old12_positive_fraction']:>6.1%}  "
            f"{row['fresh12_positive_fraction']:>8.1%}  "
            f"{row['ensemble_positive_fraction']:>9.1%}  "
            f"{row['ensemble_positive_count']:>4d}/{row['ensemble_negative_count']:<4d}      "
            f"{row['mean_positive_probe_count']:>6.2f}/24"
        )
    print(f"[validated] oriented component signs match cached binary vectors; mismatches={mismatch}")
    print(f"[saved] {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
