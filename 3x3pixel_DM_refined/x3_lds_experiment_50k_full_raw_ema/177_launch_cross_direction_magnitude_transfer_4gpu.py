"""Launch and summarize finite cross-direction magnitude transfer."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

from cross_direction_magnitude_transfer_config import *


def correlation(left, right, method):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    function = pearsonr if method == "pearson" else spearmanr
    return float(function(left, right).statistic)


def describe_pair(original, other):
    original = np.asarray(original, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    ratio = other / np.maximum(original, NSDL_EPS)
    return {
        "count": int(len(original)),
        "pearson": correlation(original, other, "pearson"),
        "spearman": correlation(original, other, "spearman"),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_median": float(np.median(ratio)),
        "other_smaller_fraction": float(np.mean(other < original)),
        "original_mean": float(np.mean(original)),
        "other_mean": float(np.mean(other)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    CDMT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CDMT_LOG_DIR / "cross_direction_magnitude_transfer_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "176_run_cross_direction_magnitude_transfer_worker.py",
                "--gpu",
                str(gpu),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(gpus)),
            ]
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            active.append((gpu, process))
            print(f"[launcher] gpu={gpu} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        while active:
            for item in list(active):
                gpu, process = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                print(f"[launcher] gpu={gpu} code={code}", flush=True)
                if code:
                    for _, other in active:
                        other.terminate()
                    raise SystemExit(code)
            if active:
                time.sleep(2)

    records = []
    timestamp_records = []
    for source_index in nsdl_datapoint_indices():
        source_dir = cdmt_source_dir(source_index)
        with open(source_dir / "result.json") as handle:
            result = json.load(handle)
        with np.load(source_dir / "magnitude_by_timestamp.npz") as archive:
            for block in result["blocks"]:
                block_index = block["block_index"]
                records.append(
                    {
                        "source": source_index,
                        "block": block_index,
                        "plus": block["source_plus_l2_mean"],
                        "minus": block["source_minus_l2_mean"],
                        "target_mean": block["target_l2_mean"],
                        "target_directions": block["target_direction_l2_means"],
                    }
                )
                timestamp_records.append(
                    {
                        "source": source_index,
                        "block": block_index,
                        "plus": archive[f"block_{block_index}_source_plus_l2"].copy(),
                        "minus": archive[f"block_{block_index}_source_minus_l2"].copy(),
                        "target_mean": archive[
                            f"block_{block_index}_target_mean_l2"
                        ].copy(),
                    }
                )

    def summarize(selection):
        plus = [record["plus"] for record in selection]
        output = {
            "source_opposite_vs_original": describe_pair(
                plus, [record["minus"] for record in selection]
            ),
            "target_mean_vs_original": describe_pair(
                plus, [record["target_mean"] for record in selection]
            ),
            "target_direction_vs_original": {},
        }
        for direction_index in range(NTCD_TARGET_DIRECTION_COUNT):
            output["target_direction_vs_original"][str(direction_index)] = describe_pair(
                plus,
                [
                    record["target_directions"][direction_index]
                    for record in selection
                ],
            )
        return output

    per_branch_timestamp_correlations = []
    for record in timestamp_records:
        per_branch_timestamp_correlations.append(
            {
                "source": record["source"],
                "block": record["block"],
                "opposite_spearman": correlation(
                    record["plus"], record["minus"], "spearman"
                ),
                "target_mean_spearman": correlation(
                    record["plus"], record["target_mean"], "spearman"
                ),
            }
        )

    fixed_timestamp_correlations = {}
    for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
        selection = [
            record for record in timestamp_records if record["block"] == block_index
        ]
        opposite_values = []
        target_values = []
        for timestamp in range(T):
            plus = [record["plus"][timestamp] for record in selection]
            opposite_values.append(
                correlation(
                    plus,
                    [record["minus"][timestamp] for record in selection],
                    "spearman",
                )
            )
            target_values.append(
                correlation(
                    plus,
                    [record["target_mean"][timestamp] for record in selection],
                    "spearman",
                )
            )
        fixed_timestamp_correlations[str(block_index)] = {
            "source_opposite_spearman_mean": float(np.nanmean(opposite_values)),
            "source_opposite_spearman_std": float(np.nanstd(opposite_values)),
            "target_mean_spearman_mean": float(np.nanmean(target_values)),
            "target_mean_spearman_std": float(np.nanstd(target_values)),
            "source_opposite_spearman_by_timestamp": opposite_values,
            "target_mean_spearman_by_timestamp": target_values,
        }

    output = {
        "definition": {
            "original": "source datapoint on trained +epsilon axis",
            "opposite": "same source datapoint on -epsilon axis",
            "other": "different target datapoint with own prompt on five independent axes",
            "magnitude": "mean over 1000 timestamps of predicted-noise delta L2",
        },
        "all_40_branches": summarize(records),
        "per_block": {},
        "per_branch_timestamp_spearman": per_branch_timestamp_correlations,
        "fixed_timestamp_cross_source_spearman": fixed_timestamp_correlations,
    }
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        selection = [record for record in records if record["block"] == block_index]
        output["per_block"][str(block_index)] = {
            "timestamp_start": timestamps[0],
            "timestamp_end": timestamps[-1],
            **summarize(selection),
        }
    CDMT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(CDMT_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)

    print("scope       comparison                 pearson  spearman  ratio  smaller", flush=True)
    rows = (("opposite", "source_opposite_vs_original"), ("target5", "target_mean_vs_original"))
    for scope, summary in [("all", output["all_40_branches"])] + [
        (f"t{value['timestamp_start']:04d}-{value['timestamp_end']:04d}", value)
        for value in output["per_block"].values()
    ]:
        for label, key in rows:
            metrics = summary[key]
            print(
                f"{scope:11s} {label:26s} "
                f"{metrics['pearson']:+.4f}  {metrics['spearman']:+.4f}  "
                f"{metrics['ratio_median']:.4f}  {metrics['other_smaller_fraction']:.3f}",
                flush=True,
            )
    timestamp_opp = np.asarray(
        [entry["opposite_spearman"] for entry in per_branch_timestamp_correlations]
    )
    timestamp_target = np.asarray(
        [entry["target_mean_spearman"] for entry in per_branch_timestamp_correlations]
    )
    print(
        "within-branch timestamp Spearman: "
        f"opposite={np.nanmean(timestamp_opp):+.4f}±{np.nanstd(timestamp_opp):.4f} | "
        f"target5={np.nanmean(timestamp_target):+.4f}±{np.nanstd(timestamp_target):.4f}",
        flush=True,
    )
    for block_index, metrics in fixed_timestamp_correlations.items():
        print(
            f"fixed-t cross-source block={block_index}: "
            f"opposite={metrics['source_opposite_spearman_mean']:+.4f}"
            f"±{metrics['source_opposite_spearman_std']:.4f} | "
            f"target5={metrics['target_mean_spearman_mean']:+.4f}"
            f"±{metrics['target_mean_spearman_std']:.4f}",
            flush=True,
        )
    print(f"[saved] {CDMT_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
