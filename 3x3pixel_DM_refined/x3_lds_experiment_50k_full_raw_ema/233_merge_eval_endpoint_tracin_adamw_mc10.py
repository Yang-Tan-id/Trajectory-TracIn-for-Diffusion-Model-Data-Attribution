"""Merge and evaluate 100x10 endpoint-loss full-AdamW TracIn."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from endpoint_tracin_adamw_mc10_config import *


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    args = parser.parse_args()
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    totals = {
        variant: {
            contraction: np.zeros((100, N_TRAIN), dtype=np.float64)
            for contraction in ETA_CONTRACTIONS
        }
        for variant in ETA_VARIANTS
    }
    for family in FAMILIES:
        query_ids = sorted(
            query_id
            for query_id, record in by_id.items()
            if record["family"] == family
        )
        covered = []
        for shard_index in range(args.timestamp_shard_count):
            root = eta_shard_root(family, shard_index, args.timestamp_shard_count)
            with open(root / "done.json") as handle:
                info = json.load(handle)
            if info["query_ids"] != query_ids:
                raise ValueError(f"query mismatch in {root}")
            covered.extend(int(value) for value in info["timestamp_indices"])
            for variant in ETA_VARIANTS:
                for contraction in ETA_CONTRACTIONS:
                    totals[variant][contraction][query_ids] += np.load(
                        root / f"{variant}_{contraction}.npy"
                    ).astype(np.float64)
        if sorted(covered) != list(range(100)):
            raise ValueError(f"{family} shards do not cover all timestamps")

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {"query_ids": list(ETA_QUERY_IDS), "variants": {}}
    for variant in ETA_VARIANTS:
        output["variants"][variant] = {}
        for contraction in ETA_CONTRACTIONS:
            method = eta_method(variant, contraction)
            scores = totals[variant][contraction]
            for query_id in ETA_QUERY_IDS:
                destination = ATTR_DIR / method / f"q{query_id:02d}"
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / "scores.npy", scores[query_id])
                atomic_json(
                    destination / "info.json",
                    {
                        "query": by_id[query_id],
                        "method": method,
                        "query_objective": "endpoint simple-loss gradient",
                        "query_timestamps": 100,
                        "query_mc": 10,
                        "train_mc": 10,
                        "query_train_noise_alignment": False,
                        "parameter_transform": "full saved-state AdamW",
                        "normalization": variant,
                        "contraction": contraction,
                    },
                )
            result = {"method": method, "targets": {}}
            print(f"\nMETHOD: {method}", flush=True)
            for metric in LDS_METRICS:
                observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
                signs = {}
                for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                    per_query = [
                        float(
                            spearmanr(
                                sign * (membership @ scores[query_id]),
                                observed[query_id],
                            ).statistic
                        )
                        for query_id in ETA_QUERY_IDS
                    ]
                    signs[sign_name] = {
                        "mean": float(np.nanmean(per_query)),
                        "std": float(np.nanstd(per_query)),
                        "per_query": per_query,
                    }
                result["targets"][metric] = signs
                print(
                    f"{metric:30s} "
                    f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                    f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                    flush=True,
                )
            output["variants"][variant][contraction] = result

    path = LDS_DIR / "endpoint_tracin_adamw_full_100t_mc10_norm4_q00_q99.json"
    atomic_json(path, output)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
