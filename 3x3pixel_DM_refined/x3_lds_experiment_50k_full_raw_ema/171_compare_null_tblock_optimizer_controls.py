"""Compare output effects from restored, fresh, and zero-gradient optimizers."""

import json

from null_tblock_cross_direction_config import *


def load_summary(optimizer_mode):
    path = ntcd_mode_summary_path(optimizer_mode)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run 170_launch_null_tblock_optimizer_controls_4gpu.py"
        )
    with open(path) as handle:
        return json.load(handle)


def main():
    summaries = {
        optimizer_mode: load_summary(optimizer_mode)
        for optimizer_mode in NTCD_OPTIMIZER_MODES
    }
    restored = summaries["restored_adamw"]
    output = {
        "optimizer_modes": list(NTCD_OPTIMIZER_MODES),
        "blocks": [],
    }
    print(
        "optimizer                      block      delta_L2      vs_restored  "
        "cross_dir_cos  parameter_delta",
        flush=True,
    )
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        restored_l2 = restored["blocks"][block_index]["metrics"][
            "delta_l2_mean"
        ]["mean"]
        block_output = {
            "block_index": block_index,
            "timestamp_start": timestamps[0],
            "timestamp_end": timestamps[-1],
            "modes": {},
        }
        for optimizer_mode in NTCD_OPTIMIZER_MODES:
            block = summaries[optimizer_mode]["blocks"][block_index]
            l2 = block["metrics"]["delta_l2_mean"]["mean"]
            cosine = block["metrics"][
                "global_off_diagonal_delta_cosine_mean"
            ]["mean"]
            parameter_delta = block.get("parameter_delta_norm", {}).get(
                "mean", float("nan")
            )
            ratio = l2 / restored_l2 if restored_l2 else float("nan")
            block_output["modes"][optimizer_mode] = {
                "delta_l2_mean": l2,
                "delta_l2_over_restored_adamw": ratio,
                "cross_direction_delta_cosine_mean": cosine,
                "parameter_delta_norm_mean": parameter_delta,
            }
            print(
                f"{optimizer_mode:30s} {timestamps[0]:04d}-{timestamps[-1]:04d}  "
                f"{l2:.6e}  {ratio:10.4f}  {cosine:+.6f}  "
                f"{parameter_delta:.6e}",
                flush=True,
            )
        output["blocks"].append(block_output)
    output_path = NTCD_CONTROL_ROOT / "optimizer_control_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
