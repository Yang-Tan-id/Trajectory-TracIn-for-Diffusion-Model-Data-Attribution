"""Merge 100-query raw/full-AdamW TracIn-DAS variants and evaluate LDS."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import TRACIN_DAS_ALL_QUERY_IDS
from tracin_das_norm_adam_config import *


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
    methods = tdna_methods()
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    all_scores = {
        transform: {
            variant: np.zeros((100, N_TRAIN), dtype=np.float64)
            for variant in TDNA_VARIANTS
        }
        for transform in TDNA_TRANSFORMS
    }

    metadata = []
    for family in FAMILIES:
        query_ids = [
            query_id
            for query_id in TRACIN_DAS_ALL_QUERY_IDS
            if by_id[query_id]["family"] == family
        ]
        covered = []
        family_scores = {
            transform: {
                variant: {
                    contraction: np.zeros(
                        (len(query_ids), N_TRAIN), dtype=np.float64
                    )
                    for contraction in TDNA_CONTRACTIONS
                }
                for variant in TDNA_VARIANTS
            }
            for transform in TDNA_TRANSFORMS
        }
        for shard_index in range(args.timestamp_shard_count):
            root = tdna_shard_root(
                family, shard_index, args.timestamp_shard_count
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            metadata.append(info)
            if info["query_ids"] != query_ids:
                raise ValueError(f"query mismatch in {root}")
            covered.extend(int(value) for value in info["timestamp_indices"])
            for transform in TDNA_TRANSFORMS:
                for variant in TDNA_VARIANTS:
                    group = f"{transform}__{variant}"
                    for contraction in TDNA_CONTRACTIONS:
                        path = root / f"{group}_{contraction}.npy"
                        family_scores[transform][variant][contraction] += np.load(
                            path
                        ).astype(np.float64)
        if sorted(covered) != list(range(100)):
            raise ValueError(f"{family} shards do not cover all 100 timestamps")

        for transform in TDNA_TRANSFORMS:
            for variant in TDNA_VARIANTS:
                for contraction in TDNA_CONTRACTIONS:
                    method = methods[transform][variant][contraction]
                    values = family_scores[transform][variant][contraction]
                    for position, query_id in enumerate(query_ids):
                        output = ATTR_DIR / method / f"q{query_id:02d}"
                        output.mkdir(parents=True, exist_ok=True)
                        np.save(output / "scores.npy", values[position])
                        atomic_json(
                            output / "info.json",
                            {
                                "query": by_id[query_id],
                                "method": method,
                                "transform": transform,
                                "normalization": variant,
                                "contraction": contraction,
                                "checkpoint_transitions": 49,
                                "timestamps": 100,
                                "parameter_projection": "CountSketch4096",
                                "train_query_noise_alignment": "same noise and t",
                                "adamw_zero_baseline_subtracted": False,
                            },
                        )
                    if contraction == "linear":
                        all_scores[transform][variant][query_ids] = values

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(TRACIN_DAS_ALL_QUERY_IDS),
        "variants": {},
    }
    for transform in TDNA_TRANSFORMS:
        result["variants"][transform] = {}
        for variant in TDNA_VARIANTS:
            result["variants"][transform][variant] = {}
            for contraction in TDNA_CONTRACTIONS:
                method = methods[transform][variant][contraction]
                scores = np.stack(
                    [
                        np.load(
                            ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
                        ).astype(np.float64)
                        for query_id in TRACIN_DAS_ALL_QUERY_IDS
                    ]
                )
                method_result = {"method": method, "targets": {}}
                print(f"\nMETHOD: {method}", flush=True)
                for metric in LDS_METRICS:
                    observed = np.load(
                        LDS_DIR / f"observed_{metric}.npy"
                    ).astype(np.float64)
                    signs = {}
                    for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                        per_query = [
                            float(
                                spearmanr(
                                    sign * (membership @ scores[query_id]),
                                    observed[query_id],
                                ).statistic
                            )
                            for query_id in TRACIN_DAS_ALL_QUERY_IDS
                        ]
                        signs[sign_name] = {
                            "mean": float(np.nanmean(per_query)),
                            "std": float(np.nanstd(per_query)),
                            "per_query": per_query,
                        }
                    method_result["targets"][metric] = signs
                    print(
                        f"{metric:30s} "
                        f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                        f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                        flush=True,
                    )
                result["variants"][transform][variant][contraction] = method_result

    output_path = LDS_DIR / "tracin_das_norm4_gradient_and_adamw_full_q00_q99.json"
    atomic_json(output_path, result)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
