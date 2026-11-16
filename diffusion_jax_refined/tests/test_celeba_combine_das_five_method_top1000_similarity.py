from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "celeba" / "script" / "combine_das_five_method_top1000_similarity.py"
METRICS = ("clip_cosine", "dino_cosine", "pixel_rmse", "lpips")


class CombineTop1000SimilarityTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_combines_common_queries_and_writes_paper_table(self) -> None:
        baseline_labels = (
            "PINT ADAMW / BOTH-L2",
            "ENDPOINT TRACIN ADAMW / RAW",
            "ENDPOINT TRACIN ADAMW / BOTH-L2",
            "RETRAC ADAMW / BOTH-L2",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.csv"
            das = root / "das.csv"
            output = root / "output"
            self.write_csv(
                baseline,
                [
                    {
                        "selection": "top",
                        "query_id": query,
                        "configuration": label,
                        "k": 1000,
                        **{metric: query + position / 10 for metric in METRICS},
                    }
                    for query in (0, 1)
                    for position, label in enumerate(baseline_labels)
                ],
            )
            self.write_csv(
                das,
                [
                    {
                        "query_id": query,
                        "configuration": "DAS EMA / lambda=200",
                        "topk": 1000,
                        **{metric: query + 0.9 for metric in METRICS},
                    }
                    for query in (0, 1)
                ],
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--baseline-csv",
                    str(baseline),
                    "--das-csv",
                    str(das),
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = (output / "report.txt").read_text(encoding="utf-8")
            self.assertIn("PAPER CONFIGURATIONS", report)
            for label in ("DAS", "PINT", "TracInCP", "GAS", "ReTrac"):
                self.assertIn(label, report)
            with (output / "per_query.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)


if __name__ == "__main__":
    unittest.main()
