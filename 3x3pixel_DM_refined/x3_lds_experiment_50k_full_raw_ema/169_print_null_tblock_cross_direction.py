"""Print per-source results for the timestamp-block direction experiment."""

import json

from null_tblock_cross_direction_config import *


def main():
    for source_index in nsdl_datapoint_indices():
        path = ntcd_source_dir(source_index) / "result.json"
        with open(path) as handle:
            result = json.load(handle)
        print(
            f"\nsource={source_index:05d} target={result['target_datapoint_index']:05d} "
            f"target-axis cosines={['%+.3f' % x for x in result['target_direction_cosine_to_source']]}"
        )
        for block in result["blocks"]:
            metric = block["metrics"]
            print(
                f"  t={block['timestamp_start']:04d}-{block['timestamp_end']:04d} "
                f"L2={metric['delta_l2_mean']:.6e} "
                f"RMSE={metric['delta_rmse_mean']:.6e} "
                f"cross-dir-cos={metric['global_off_diagonal_delta_cosine_mean']:+.6f} "
                f"noise/delta-corr={metric['noise_cosine_vs_delta_cosine_correlation']:+.4f}"
            )


if __name__ == "__main__":
    main()
