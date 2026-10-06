"""Merge endpoint100 responses, save every timestamp score, and evaluate LDS."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from endpoint20_meanloss_pairing_config import *
from exp_config import DAS_TIMESTEPS


METHOD = (
    "tracin_das_endpoint100_inverse_noise_10ckpt_100t_"
    "per_timestamp_aligned_adamw_full_next_delta_projected4096_raw"
)


def shard_root(checkpoint_shard_index, checkpoint_shard_count):
    return (
        ATTR_DIR
        / "_tracin_das_endpoint100_inverse_noise_per_timestamp_10q_shards"
        / f"checkpoint_shard_{int(checkpoint_shard_index):02d}_of_"
          f"{int(checkpoint_shard_count):02d}"
    )


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-shard-count", type=int, default=4)
    args = parser.parse_args()

    query_ids = tuple(range(10))
    timesteps = tuple(DAS_TIMESTEPS)
    response = np.zeros(
        (len(query_ids), len(timesteps), N_TRAIN), dtype=np.float32
    )
    covered = []
    resolved = None
    for shard_index in range(args.checkpoint_shard_count):
        root = shard_root(shard_index, args.checkpoint_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        covered.extend(int(value) for value in info["checkpoint_pairs"])
        shard_resolved = tuple(int(value) for value in info["resolved_trajectory_timesteps"])
        if resolved is None:
            resolved = shard_resolved
        elif resolved != shard_resolved:
            raise ValueError("resolved trajectory timestamps differ across shards")
        with np.load(root / "partial_scores.npz") as partial:
            response += partial[
                "per_timestamp_aligned__timestamp_response"
            ].astype(np.float32)
    if sorted(covered) != sorted(NPA_CHECKPOINT_PAIRS):
        raise ValueError(f"checkpoint coverage mismatch: {sorted(covered)}")

    scores_by_timestamp = np.square(response, dtype=np.float32)
    aggregate_scores = scores_by_timestamp.mean(axis=1, dtype=np.float32)
    for position, query_id in enumerate(query_ids):
        output = ATTR_DIR / METHOD / f"q{query_id:02d}"
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "scores_by_timestamp.npy", scores_by_timestamp[position])
        np.save(output / "scores.npy", aggregate_scores[position])
        atomic_json(
            output / "info.json",
            {
                "method": METHOD,
                "query_id": query_id,
                "timesteps": list(timesteps),
                "resolved_trajectory_timesteps": list(resolved),
                "scores_by_timestamp_shape": list(scores_by_timestamp[position].shape),
                "aggregate": "mean over timestamp squared checkpoint-summed response",
                "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
            },
        )

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float32)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(query_ids)
        ]
        for metric in LDS_METRICS
    }

    # One BLAS call computes all q,t subset predictions.
    flat_scores = scores_by_timestamp.reshape(
        len(query_ids) * len(timesteps), N_TRAIN
    )
    per_t_prediction = (membership @ flat_scores.T).T.reshape(
        len(query_ids), len(timesteps), membership.shape[0]
    )
    aggregate_prediction = membership @ aggregate_scores.T

    result = {
        "method": METHOD,
        "query_ids": list(query_ids),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timesteps": list(timesteps),
        "resolved_trajectory_timesteps": list(resolved),
        "score_definition": "square of checkpoint-summed per-t response",
        "aggregate": {},
        "per_timestamp": [],
    }
    lines = [
        "ENDPOINT100 PER-TIMESTAMP ALIGNED TRACIN-DAS",
        "10 checkpoint pairs; 100 timestamps; full AdamW; projected4096; sign=-1",
        "",
        "[all-100 timestamp mean]",
    ]
    for metric in LDS_METRICS:
        per_query = np.asarray(
            [
                spearmanr(-aggregate_prediction[:, q], observed[metric][q]).statistic
                for q in range(len(query_ids))
            ],
            dtype=np.float64,
        )
        result["aggregate"][metric] = {
            "mean": float(np.nanmean(per_query)),
            "std": float(np.nanstd(per_query)),
            "per_query": per_query.tolist(),
        }
        lines.append(
            f"{metric:34s} {np.nanmean(per_query):+.6f} "
            f"+/- {np.nanstd(per_query):.6f}"
        )

    lines.extend(["", "[each timestamp]", "timestep " + " ".join(
        f"{metric:>24s}" for metric in LDS_METRICS
    )])
    for timestamp_position, timestep in enumerate(timesteps):
        entry = {
            "timestamp_position": timestamp_position,
            "timestep": int(timestep),
            "resolved_trajectory_timestep": int(resolved[timestamp_position]),
            "targets": {},
        }
        cells = []
        for metric in LDS_METRICS:
            per_query = np.asarray(
                [
                    spearmanr(
                        -per_t_prediction[q, timestamp_position],
                        observed[metric][q],
                    ).statistic
                    for q in range(len(query_ids))
                ],
                dtype=np.float64,
            )
            entry["targets"][metric] = {
                "mean": float(np.nanmean(per_query)),
                "std": float(np.nanstd(per_query)),
                "per_query": per_query.tolist(),
            }
            cells.append(f"{np.nanmean(per_query):+10.6f}")
        result["per_timestamp"].append(entry)
        lines.append(f"t={int(timestep):04d}  " + " ".join(cells))

    json_path = LDS_DIR / f"{METHOD}_q00_q09.json"
    text_path = LDS_DIR / f"{METHOD}_q00_q09.txt"
    atomic_json(json_path, result)
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")
    print(f"[saved] {ATTR_DIR / METHOD / 'qXX' / 'scores_by_timestamp.npy'}")


if __name__ == "__main__":
    main()
