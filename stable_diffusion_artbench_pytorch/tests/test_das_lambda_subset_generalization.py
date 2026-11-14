import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from sd_artbench.analyze_das_lambda_subset_generalization import build_report


class DasLambdaSubsetGeneralizationTest(unittest.TestCase):
    def test_reports_subset_and_leave_one_out_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "per_query.csv"
            fields = (
                "query_index",
                "configuration",
                "clip_cosine",
                "dino_cosine",
                "pixel_rmse",
                "lpips",
            )
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for query in range(4):
                    for damping in (100.0, 1000.0):
                        best = damping == 1000.0
                        writer.writerow(
                            {
                                "query_index": query,
                                "configuration": f"DAS lambda={damping:g}",
                                "clip_cosine": 0.8 if best else 0.7,
                                "dino_cosine": 0.8 if best else 0.7,
                                "pixel_rmse": 0.2 if best else 0.3,
                                "lpips": 0.2 if best else 0.3,
                            }
                        )
            args = argparse.Namespace(
                per_query_csv=source,
                output=root / "report.txt",
                expected_queries=4,
                num_subsets=2,
                partition_seed=2026,
                fixed_lambda=1000.0,
            )
            report = build_report(args)
            self.assertIn("Fixed lambda subset-best count: 2/2", report)
            self.assertIn("Fixed lambda LOO-selection count: 2/2", report)
            self.assertIn("Full-query best lambda: 1000", report)


if __name__ == "__main__":
    unittest.main()
