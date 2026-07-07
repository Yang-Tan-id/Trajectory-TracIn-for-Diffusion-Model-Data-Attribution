from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.aggregate_lds_by_seed import aggregate_group


class TestAggregateLdsBySeed(unittest.TestCase):
    def test_aggregate_group_writes_summary_and_scatter_svgs(self):
        with tempfile.TemporaryDirectory() as tmp:
            group = Path(tmp) / "query_horse" / "initial_seed_42" / "lds" / "traj_tracin" / "simple_loss"
            for seed in (1, 2):
                seed_dir = group / f"m_50_k_8000_seed_{seed}"
                seed_dir.mkdir(parents=True)
                with (seed_dir / "lds_results.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["subset_id", "pred_sum_tau", "true_f", "checkpoint"],
                    )
                    writer.writeheader()
                    for subset_id in range(3):
                        writer.writerow(
                            {
                                "subset_id": subset_id,
                                "pred_sum_tau": seed * 10 + subset_id,
                                "true_f": seed * 0.1 + subset_id * 0.01,
                                "checkpoint": f"/fake/m_50_k_8000_seed_{seed}/subset_{subset_id:04d}/ckpt",
                            }
                        )

            out = aggregate_group(group, model_glob="m_50_k_8000_seed_*", output_name="aggregate_test")
            self.assertIsNotNone(out)
            assert out is not None
            self.assertTrue((out / "per_seed_summary.csv").is_file())
            self.assertTrue((out / "per_seed_summary.json").is_file())
            self.assertTrue((out / "all_seed_points.csv").is_file())
            self.assertTrue((out / "per_seed_scatter_grid.svg").is_file())
            self.assertTrue((out / "all_points_scatter.svg").is_file())
            payload = json.loads((out / "per_seed_summary.json").read_text())
            self.assertEqual(payload["num_seeds"], 2)
            self.assertEqual(payload["num_points"], 6)
            self.assertIn("lds_percent_mean", payload)


if __name__ == "__main__":
    unittest.main()
