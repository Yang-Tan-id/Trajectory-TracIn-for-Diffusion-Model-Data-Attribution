"""Evaluate the endpoint20 experiment one timestamp at a time."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from endpoint20_meanloss_pairing_config import *


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


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

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(E20_QUERY_IDS)
        ]
        for metric in LDS_METRICS
    }

    result = {
        "mode": args.mode,
        "query_ids": list(E20_QUERY_IDS),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_positions": list(E20_TIMESTAMP_POSITIONS),
        "timesteps": list(E20_TIMESTEPS),
        "score_definition": "square of checkpoint-summed response at one timestamp",
        "per_timestamp": [],
    }
    lines = [
        "ENDPOINT20: LDS OF EACH TIMESTAMP",
        f"mode={args.mode}; sign=-1; queries=q00-q09",
        "",
    ]
    header = "timestamp " + " ".join(f"{metric:>28s}" for metric in LDS_METRICS)
    lines.append(header)

    for local_index, timestep in enumerate(E20_TIMESTEPS):
        # The common 1/20 factor in the full timestamp mean does not affect
        # Spearman correlation, so the isolated timestamp score is simply
        # the squared checkpoint-summed response.
        score = np.square(response[:, local_index, :])
        prediction = membership @ score.T
        entry = {
            "timestamp_position": int(E20_TIMESTAMP_POSITIONS[local_index]),
            "timestep": int(timestep),
            "targets": {},
        }
        cells = []
        for metric in LDS_METRICS:
            per_query = np.asarray(
                [
                    spearmanr(-prediction[:, q], observed[metric][q]).statistic
                    for q in range(len(E20_QUERY_IDS))
                ],
                dtype=np.float64,
            )
            entry["targets"][metric] = {
                "mean": float(np.nanmean(per_query)),
                "std": float(np.nanstd(per_query)),
                "per_query": per_query.tolist(),
            }
            cells.append(
                f"{np.nanmean(per_query):+10.6f}+/-{np.nanstd(per_query):.6f}"
            )
        result["per_timestamp"].append(entry)
        line = f"t={int(timestep):04d}  " + " ".join(cells)
        lines.append(line)
        print(line, flush=True)

    stem = f"tracin_das_endpoint20_{args.mode}_per_timestamp_lds_q00_q09"
    json_path = LDS_DIR / f"{stem}.json"
    text_path = LDS_DIR / f"{stem}.txt"
    atomic_json(json_path, result)
    text_path.write_text("\n".join(lines) + "\n")
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
