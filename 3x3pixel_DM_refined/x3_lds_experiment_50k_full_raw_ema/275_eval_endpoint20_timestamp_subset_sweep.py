"""Evaluate nested and evenly-spaced timestamp subsets from endpoint20 shards."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from endpoint20_meanloss_pairing_config import *


PREFIX_COUNTS = (1, 2, 3, 5, 10, 15, 20)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def paired_test(candidate, baseline):
    difference = candidate - baseline
    finite = difference[np.isfinite(difference)]
    try:
        pvalue = float(wilcoxon(finite, alternative="two-sided").pvalue)
    except ValueError:
        pvalue = 1.0
    return {
        "mean_difference_vs_all20": float(np.nanmean(difference)),
        "wilcoxon_two_sided_p_vs_all20": pvalue,
        "wins_vs_all20": int(np.count_nonzero(difference > 0)),
        "query_count": int(len(finite)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-shard-count", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=E20_MODES,
        default="per_timestamp_aligned",
    )
    args = parser.parse_args()

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

    # These subsets all reuse exactly the same cached responses and projection.
    selections = {
        f"prefix_{count:02d}": tuple(range(count)) for count in PREFIX_COUNTS
    }
    for count in (2, 3, 5, 10):
        selections[f"even_{count:02d}"] = tuple(
            np.linspace(0, len(E20_TIMESTEPS) - 1, count, dtype=int).tolist()
        )

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(E20_QUERY_IDS)
        ]
        for metric in LDS_METRICS
    }
    values = {}
    result = {
        "mode": args.mode,
        "query_ids": list(E20_QUERY_IDS),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "available_timesteps": list(E20_TIMESTEPS),
        "score_definition": "mean_t square(checkpoint-summed response)",
        "selections": {},
    }

    for name, indices in selections.items():
        score = np.square(response[:, indices, :]).mean(axis=1)
        prediction = membership @ score.T
        values[name] = {}
        entry = {
            "local_indices": list(indices),
            "timesteps": [int(E20_TIMESTEPS[index]) for index in indices],
            "targets": {},
        }
        for metric in LDS_METRICS:
            per_query = np.asarray(
                [
                    spearmanr(-prediction[:, q], observed[metric][q]).statistic
                    for q in range(len(E20_QUERY_IDS))
                ],
                dtype=np.float64,
            )
            values[name][metric] = per_query
            entry["targets"][metric] = {
                "mean": float(np.nanmean(per_query)),
                "std": float(np.nanstd(per_query)),
                "per_query": per_query.tolist(),
            }
        result["selections"][name] = entry

    baseline_name = "prefix_20"
    lines = [
        "ENDPOINT20 TIMESTAMP-SUBSET SWEEP",
        f"mode={args.mode}; sign=-1; queries=q00-q09",
        "All variants reuse the same cached responses and projection.",
        "",
    ]
    for name, indices in selections.items():
        timesteps = [int(E20_TIMESTEPS[index]) for index in indices]
        lines.append(f"[{name}] timestamps={timesteps}")
        lines.append(
            "target                              mean       std      "
            "delta-vs-20       p       wins"
        )
        for metric in LDS_METRICS:
            test = paired_test(values[name][metric], values[baseline_name][metric])
            result["selections"][name]["targets"][metric]["vs_all20"] = test
            mean = result["selections"][name]["targets"][metric]["mean"]
            std = result["selections"][name]["targets"][metric]["std"]
            lines.append(
                f"{metric:34s} {mean:+.6f} {std:.6f} "
                f"{test['mean_difference_vs_all20']:+.6f} "
                f"{test['wilcoxon_two_sided_p_vs_all20']:.6g} "
                f"{test['wins_vs_all20']:02d}/10"
            )
        lines.append("")

    stem = f"tracin_das_endpoint20_{args.mode}_timestamp_subset_sweep_q00_q09"
    json_path = LDS_DIR / f"{stem}.json"
    text_path = LDS_DIR / f"{stem}.txt"
    atomic_json(json_path, result)
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
