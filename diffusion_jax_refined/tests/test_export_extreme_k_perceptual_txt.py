from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "script/export_celeba_artbench_extreme_k_txt.py"
)
SPEC = importlib.util.spec_from_file_location("export_extreme_k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExportExtremeKTest(unittest.TestCase):
    def test_exports_report_and_all_per_query_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "result"
            source.mkdir()
            (source / "report.txt").write_text("AGGREGATE\n", encoding="utf-8")
            with (source / "per_query.csv").open("w", newline="") as handle:
                fields = [
                    "selection", "query_index", "method", "transform", "k",
                    "clip_cosine", "dino_cosine", "pixel_rmse", "lpips",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for selection in ("top", "bottom"):
                    for transform in ("raw_gradient", "adamw_full"):
                        for method in range(5):
                            writer.writerow(
                                {
                                    "selection": selection,
                                    "query_index": 0,
                                    "method": f"method-{method}",
                                    "transform": transform,
                                    "k": 100,
                                    "clip_cosine": 0.1,
                                    "dino_cosine": 0.2,
                                    "pixel_rmse": 0.3,
                                    "lpips": 0.4,
                                }
                            )
            with (source / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["metric", "mean"])
                writer.writeheader()
                writer.writerow({"metric": "clip_cosine", "mean": 0.1})

            output = Path(temporary) / "combined.txt"
            MODULE.export_one("artbench", 100, source, output, 1)
            text = output.read_text()
            self.assertIn("AGGREGATE REPORT", text)
            self.assertIn("PER-QUERY VALUES (READABLE TABLE)", text)
            self.assertIn("PER-QUERY VALUES (CSV", text)
            self.assertIn("SUMMARY STATISTICS (CSV)", text)
            self.assertEqual(text.count("method-0"), 8)


if __name__ == "__main__":
    unittest.main()
