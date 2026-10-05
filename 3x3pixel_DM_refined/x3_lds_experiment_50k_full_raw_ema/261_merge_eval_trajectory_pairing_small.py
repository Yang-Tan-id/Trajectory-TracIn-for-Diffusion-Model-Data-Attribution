"""Merge trajectory-aligned/independent small controlled comparison."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, N_TRAIN
from noise_pairing_ablation_config import NPA_TIMESTAMP_GROUPS


MODES = ("trajectory_aligned", "trajectory_independent")
ROOT_MODES = {"trajectory_aligned": "aligned", "trajectory_independent": "independent"}
CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
QUERY_IDS = tuple(range(10))
ROOT = ATTR_DIR / "_tracin_das_trajectory_pairing_small_shards"
BASELINE = LDS_DIR / "tracin_das_noise_pairing_ablation_10ckpt_20t_mc10_q00_q09.json"


def method_name(mode, contraction, group):
    return (
        "tracin_das_pure_trajectory_direction_10ckpt_20t_mc1_"
        f"adamw_full_{mode}_next_delta_projected4096_raw_{contraction}_{group}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp-shard-count", type=int, default=4)
    args = ap.parse_args()
    totals = {
        mode: {
            contraction: {
                group: np.zeros((len(QUERY_IDS), N_TRAIN), dtype=np.float64)
                for group in NPA_TIMESTAMP_GROUPS
            }
            for contraction in CONTRACTIONS
        }
        for mode in MODES
    }
    for mode in MODES:
        root_mode = ROOT_MODES[mode]
        covered = []
        for shard in range(args.timestamp_shard_count):
            root = ROOT / root_mode / f"shard_{shard:02d}_of_{args.timestamp_shard_count:02d}"
            with open(root / "done.json") as handle:
                info = json.load(handle)
            covered.extend(int(value) for value in info["timestamp_indices"])
            with np.load(root / "partial_scores.npz") as values:
                for contraction in CONTRACTIONS:
                    for group in NPA_TIMESTAMP_GROUPS:
                        totals[mode][contraction][group] += values[
                            f"{contraction}__{group}"
                        ].astype(np.float64)
        # 'all' duplicates q1-q4, so compare against its unique timestamp set.
        if sorted(set(covered)) != sorted(set(NPA_TIMESTAMP_GROUPS["all"])):
            raise ValueError(f"timestamp coverage mismatch for {mode}: {covered}")

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {"query_ids": list(QUERY_IDS), "results": {}, "six_way_comparison": {}}
    print("TRAJECTORY-DIRECTION PAIRING SMALL TEST (sign=-1)")
    for mode in MODES:
        result["results"][mode] = {}
        for contraction in CONTRACTIONS:
            result["results"][mode][contraction] = {}
            for group in NPA_TIMESTAMP_GROUPS:
                method = method_name(mode, contraction, group)
                score = totals[mode][contraction][group]
                for query_id in QUERY_IDS:
                    output = ATTR_DIR / method / f"q{query_id:02d}"
                    output.mkdir(parents=True, exist_ok=True)
                    np.save(output / "scores.npy", score[query_id])
                prediction = membership @ score.T
                entry = {"method": method, "targets": {}}
                for metric in LDS_METRICS:
                    observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                        np.float64
                    )[list(QUERY_IDS)]
                    per_query = [
                        float(spearmanr(-prediction[:, q], observed[q]).statistic)
                        for q in range(len(QUERY_IDS))
                    ]
                    entry["targets"][metric] = {
                        "negative": {
                            "mean": float(np.nanmean(per_query)),
                            "std": float(np.nanstd(per_query)),
                            "per_query": per_query,
                        }
                    }
                result["results"][mode][contraction][group] = entry

    with open(BASELINE) as handle:
        baseline = json.load(handle)
    controls = ("aligned", "cyclic", "random_permutation", "independent")
    for contraction in CONTRACTIONS:
        print(f"\n{contraction} / all timestamps")
        result["six_way_comparison"][contraction] = {}
        for metric in LDS_METRICS:
            aligned = result["results"]["trajectory_aligned"][contraction]["all"]["targets"][metric]["negative"]
            independent = result["results"]["trajectory_independent"][contraction]["all"]["targets"][metric]["negative"]
            values = {
                "trajectory_aligned": aligned["mean"],
                "trajectory_independent": independent["mean"],
            }
            for control in controls:
                per_query = baseline["results"][control]["raw"][contraction]["all"]["targets"][metric]["negative"]["per_query"][:10]
                values[control] = float(np.nanmean(per_query))
            result["six_way_comparison"][contraction][metric] = values
            print(
                f"{metric:30s} traj-A={values['trajectory_aligned']:+.4f} "
                f"traj-I={values['trajectory_independent']:+.4f} | "
                f"A={values['aligned']:+.4f} C={values['cyclic']:+.4f} "
                f"R={values['random_permutation']:+.4f} I={values['independent']:+.4f}"
            )

    output = LDS_DIR / "tracin_das_trajectory_direction_pairing_small_q00_q09.json"
    temporary = output.with_suffix(".tmp.json")
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2)
    temporary.replace(output)
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
