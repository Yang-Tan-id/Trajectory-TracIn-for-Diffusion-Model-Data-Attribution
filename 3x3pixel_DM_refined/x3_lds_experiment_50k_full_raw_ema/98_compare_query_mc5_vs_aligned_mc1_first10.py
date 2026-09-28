"""Compare q00-q09 endpoint query-MC5/train-MC1 against legacy aligned MC1."""

import json

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


QUERY_IDS = TRACIN_DAS_QUERY_IDS
TARGETS = {
    "simple_loss": ("simple_loss_ema", "simple_loss_raw"),
    "traj_ref": ("traj_ref_ema", "traj_ref_raw"),
    "endpoint_deviation": ("endpoint_deviation_ema", "endpoint_deviation_raw"),
    "trajectory_state_mse": ("trajectory_state_mse_ema", "trajectory_state_mse_raw"),
}


def load_scores(method):
    return np.stack(
        [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(
                np.float64
            )
            for query_id in QUERY_IDS
        ],
        axis=0,
    )


def correlations(scores, membership, observed):
    predictions = membership @ scores.T
    positive = np.asarray(
        [
            spearmanr(predictions[:, position], observed[query_id]).statistic
            for position, query_id in enumerate(QUERY_IDS)
        ],
        dtype=np.float64,
    )
    return {"positive": positive, "negative": -positive}


def stats(values):
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "per_query": values.tolist(),
    }


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    old_methods = tracin_das_methods(
        "checkpoint", "projected4096", "aligned", 1
    )
    new_methods = tracin_das_methods(
        "checkpoint", "projected4096", "independent-mc1", 5
    )
    output = {
        "query_ids": list(QUERY_IDS),
        "query_count": len(QUERY_IDS),
        "old": {
            "query_mc": 1,
            "train_mc": 1,
            "query_train_noise_aligned": True,
        },
        "new": {
            "query_mc": 5,
            "train_mc": 1,
            "query_train_noise_aligned": False,
        },
        "checkpoint_count": 50,
        "transition_count": 49,
        "timestamp_count": 100,
        "results": {},
    }
    lines = [
        "q00-q09 comparison: legacy aligned MC1 versus query-MC5/independent-train-MC1",
        "old = query MC1, train MC1, same noise",
        "new = query MC5, train MC1, independent query/train noise",
        "delta = new - old at the same LDS sign",
        "STD = population std across ten queries (ddof=0)",
    ]

    for contraction in ("linear", "termwise_squared", "timestamp_sum_squared"):
        old_method = old_methods[contraction]
        new_method = new_methods[contraction]
        old_scores = load_scores(old_method)
        new_scores = load_scores(new_method)
        contraction_result = {
            "old_method": old_method,
            "new_method": new_method,
            "targets": {},
        }
        lines.extend(["", "=" * 100, f"CONTRACTION: {contraction}"])
        for target_name, metrics in TARGETS.items():
            target_result = {}
            for metric in metrics:
                observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                    np.float64
                )
                old = correlations(old_scores, membership, observed)
                new = correlations(new_scores, membership, observed)
                signs = {}
                lines.extend(["", f"  METRIC: {metric}"])
                for sign_name in ("negative", "positive"):
                    delta = new[sign_name] - old[sign_name]
                    signs[sign_name] = {
                        "old": stats(old[sign_name]),
                        "new": stats(new[sign_name]),
                        "delta": stats(delta),
                        "better_queries": int(np.sum(delta > 0)),
                        "worse_queries": int(np.sum(delta < 0)),
                    }
                    lines.append(
                        f"    sign={sign_name:<8s} "
                        f"old={np.nanmean(old[sign_name]):+.6f}±{np.nanstd(old[sign_name]):.6f}  "
                        f"new={np.nanmean(new[sign_name]):+.6f}±{np.nanstd(new[sign_name]):.6f}  "
                        f"delta={np.nanmean(delta):+.6f}±{np.nanstd(delta):.6f}  "
                        f"better={int(np.sum(delta > 0))}/10"
                    )
                target_result[metric] = signs

                # The historically useful orientation is sign=-1; print its
                # ten paired values for direct query-by-query inspection.
                lines.append("    qid      old(-1)      new(-1)       delta")
                negative_delta = new["negative"] - old["negative"]
                for position, query_id in enumerate(QUERY_IDS):
                    lines.append(
                        f"    q{query_id:02d}   {old['negative'][position]:+10.6f}  "
                        f"{new['negative'][position]:+10.6f}  "
                        f"{negative_delta[position]:+10.6f}"
                    )
            contraction_result["targets"][target_name] = target_result
        output["results"][contraction] = contraction_result

    json_path = LDS_DIR / "query_mc5_independent_vs_aligned_mc1_q00_q09.json"
    text_path = LDS_DIR / "query_mc5_independent_vs_aligned_mc1_q00_q09.txt"
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    text_path.write_text("\n".join(lines) + "\n")
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
