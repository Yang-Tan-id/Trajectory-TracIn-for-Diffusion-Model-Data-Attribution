"""Mix 50-checkpoint/100-timestamp TracIn scores 50/50 with aligned DAS λ=100."""

import json

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


DAS_METHOD = "das_ema_aligned_noise"
DAS_LAMBDA = 100.0
MIX_WEIGHT = 0.5
TARGETS = {
    "simple_loss": ("simple_loss_ema", "simple_loss_raw"),
    "traj_ref": ("traj_ref_ema", "traj_ref_raw"),
    "endpoint_deviation": ("endpoint_deviation_ema", "endpoint_deviation_raw"),
    "trajectory_state_mse": ("trajectory_state_mse_ema", "trajectory_state_mse_raw"),
}


def lambda_tag(value):
    return str(float(value)).replace(".", "p")


def standardize_rows(values):
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    if np.any(scales <= 0):
        bad = np.flatnonzero(scales[:, 0] <= 0).tolist()
        raise ValueError(f"zero score standard deviation for query positions {bad}")
    return (values - means) / scales


def evaluate(scores, membership, observed):
    predictions = membership @ scores.T
    result = {}
    for target_name, metrics in TARGETS.items():
        target_result = {}
        for metric in metrics:
            signs = {}
            positive = np.asarray(
                [
                    spearmanr(
                        predictions[:, position], observed[metric][query_id]
                    ).statistic
                    for position, query_id in enumerate(
                        TRACIN_DAS_FIRST99_QUERY_IDS
                    )
                ],
                dtype=np.float64,
            )
            for sign_name, values in (
                ("positive", positive),
                ("negative", -positive),
            ):
                signs[sign_name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_query": values.tolist(),
                }
            target_result[metric] = signs
        result[target_name] = target_result
    return result


def save_score_bank(method, scores, contraction, normalization):
    for position, query_id in enumerate(TRACIN_DAS_FIRST99_QUERY_IDS):
        output = ATTR_DIR / method / f"q{query_id:02d}"
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "scores.npy", scores[position])
        with open(output / "info.json", "w") as handle:
            json.dump(
                {
                    "method": method,
                    "query_id": int(query_id),
                    "mix": {
                        "tracin_weight": MIX_WEIGHT,
                        "das_weight": MIX_WEIGHT,
                        "normalization": normalization,
                    },
                    "tracin": {
                        "checkpoint_count": 50,
                        "transition_count": 49,
                        "timestamp_count": 100,
                        "contraction": contraction,
                        "learning_rate_source": "source checkpoint saved eta",
                    },
                    "das": {
                        "method": DAS_METHOD,
                        "lambda": DAS_LAMBDA,
                        "noise_alignment": "query/train aligned",
                    },
                },
                handle,
                indent=2,
            )


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        for metrics in TARGETS.values()
        for metric in metrics
    }
    das_scores = np.stack(
        [
            np.load(
                ATTR_DIR
                / DAS_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(DAS_LAMBDA)}"
                / "scores.npy"
            ).astype(np.float64)
            for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
        ],
        axis=0,
    )

    output = {
        "query_ids": list(TRACIN_DAS_FIRST99_QUERY_IDS),
        "mix_weights": {"tracin": MIX_WEIGHT, "das": MIX_WEIGHT},
        "das_method": DAS_METHOD,
        "das_lambda": DAS_LAMBDA,
        "checkpoint_count": 50,
        "transition_count": 49,
        "timestamp_count": 100,
        "results": {},
    }
    text_lines = [
        "50/50 mixture: 50-checkpoint/100-timestamp TracIn + aligned DAS λ=100",
        "queries: q00-q98",
        "raw_mix: 0.5*TracIn + 0.5*DAS",
        "zscore_mix: per-query score standardization, then 0.5/0.5",
        "STD: population std across queries (ddof=0)",
    ]

    das_result = evaluate(das_scores, membership, observed)
    for contraction, tracin_method in tracin_das_checkpoint_lr_timestamp_methods(
        100
    ).items():
        tracin_scores = np.stack(
            [
                np.load(
                    ATTR_DIR / tracin_method / f"q{query_id:02d}" / "scores.npy"
                ).astype(np.float64)
                for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
            ],
            axis=0,
        )
        raw_mix = MIX_WEIGHT * tracin_scores + MIX_WEIGHT * das_scores
        zscore_mix = (
            MIX_WEIGHT * standardize_rows(tracin_scores)
            + MIX_WEIGHT * standardize_rows(das_scores)
        )
        raw_method = (
            f"mix50_tracin_50ckpt_100timestamp_{contraction}_"
            "das_aligned_lambda100_raw"
        )
        zscore_method = (
            f"mix50_tracin_50ckpt_100timestamp_{contraction}_"
            "das_aligned_lambda100_zscore"
        )
        save_score_bank(raw_method, raw_mix, contraction, "none")
        save_score_bank(
            zscore_method,
            zscore_mix,
            contraction,
            "per-query z-score across 50000 training points",
        )
        variants = {
            "tracin": evaluate(tracin_scores, membership, observed),
            "das_aligned_lambda100": das_result,
            "raw_mix": evaluate(raw_mix, membership, observed),
            "zscore_mix": evaluate(zscore_mix, membership, observed),
        }
        output["results"][contraction] = {
            "tracin_method": tracin_method,
            "raw_mix_method": raw_method,
            "zscore_mix_method": zscore_method,
            "variants": variants,
        }
        text_lines.extend(["", "=" * 96, f"CONTRACTION: {contraction}"])
        for metrics in TARGETS.values():
            for metric in metrics:
                text_lines.append(f"  {metric}")
                for variant_name, result in variants.items():
                    signs = next(
                        target[metric]
                        for target in result.values()
                        if metric in target
                    )
                    text_lines.append(
                        f"    {variant_name:24s} "
                        f"-1={signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f}  "
                        f"+1={signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}"
                    )

    json_path = LDS_DIR / "mix50_tracin_50ckpt_100timestamp_with_aligned_das_lambda100.json"
    text_path = LDS_DIR / "mix50_tracin_50ckpt_100timestamp_with_aligned_das_lambda100.txt"
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    text_path.write_text("\n".join(text_lines) + "\n")
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
