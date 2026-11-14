import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from diffusion_jax_refined.celeba.script.analyze_das_lambda_lds_subset_generalization import (
    build_report,
)


class CelebaDasLambdaLdsSubsetGeneralizationTest(unittest.TestCase):
    def test_target_specific_best_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "lds.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "query_id",
                        "lambda",
                        "target_function",
                        "lds_percent",
                    ),
                )
                writer.writeheader()
                for query in range(4):
                    for damping in (100.0, 200.0):
                        writer.writerow(
                            {
                                "query_id": query,
                                "lambda": damping,
                                "target_function": "simple_loss_50x1",
                                "lds_percent": 20.0 if damping == 200.0 else 10.0,
                            }
                        )
            args = argparse.Namespace(
                lds_csv=source,
                output=root / "report.txt",
                expected_queries=4,
                num_subsets=2,
                partition_seed=2026,
                fixed_lambda=None,
            )
            report = build_report(args)
            self.assertIn("Full-query best lambda=200; fixed lambda=200", report)
            self.assertIn("Fixed lambda subset-best count: 2/2", report)
            self.assertIn("Fixed lambda LOO-selection count: 2/2", report)


if __name__ == "__main__":
    unittest.main()
