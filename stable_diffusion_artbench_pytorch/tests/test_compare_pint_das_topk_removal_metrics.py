import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from sd_artbench.compare_pint_das_topk_removal_metrics import build_report


class PintDasRemovalComparisonTest(unittest.TestCase):
    def test_merges_shards_and_compares_all_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            fields = {
                "trajectory": (
                    "query_index", "method", "endpoint_mse", "endpoint_rmse",
                    "trajectory_mse", "trajectory_rmse",
                ),
                "perceptual": (
                    "query_index", "method", "pixel_l2", "clip_distance",
                    "lpips", "dino_distance",
                ),
            }

            def write(path, names, method, queries, value):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=names)
                    writer.writeheader()
                    for query in queries:
                        row = {name: value for name in names}
                        row.update(query_index=query, method=method)
                        writer.writerow(row)

            write(
                baseline / "endpoint_trajectory_metrics.csv", fields["trajectory"],
                "endpoint_pollute_query_train_l2", range(4), 2.0,
            )
            write(
                baseline / "perceptual_metrics.csv", fields["perceptual"],
                "endpoint_pollute_query_train_l2", range(4), 2.0,
            )
            trajectory_shards, perceptual_shards = [], []
            for number, queries in enumerate((range(2), range(2, 4))):
                trajectory = root / f"trajectory_{number}.csv"
                perceptual = root / f"perceptual_{number}.csv"
                write(trajectory, fields["trajectory"], "das_lambda_1000", queries, 1.0)
                write(perceptual, fields["perceptual"], "das_lambda_1000", queries, 1.0)
                trajectory_shards.append(trajectory)
                perceptual_shards.append(perceptual)
            args = argparse.Namespace(
                baseline_dir=baseline,
                das_trajectory_csv=trajectory_shards,
                das_perceptual_csv=perceptual_shards,
                query_ids=list(range(4)),
                output=root / "report.txt",
                per_query_output=None,
                pint_method="endpoint_pollute_query_train_l2",
                das_method="das_lambda_1000",
            )
            report = build_report(args)
            self.assertIn("ARTBENCH Q0-Q3", report)
            self.assertIn("Endpoint RMSE", report)
            self.assertIn("DINO distance", report)
            self.assertIn("4/4", report)
            self.assertIn("PER-QUERY VALUES", report)
            self.assertIn("Q3", report)


if __name__ == "__main__":
    unittest.main()
