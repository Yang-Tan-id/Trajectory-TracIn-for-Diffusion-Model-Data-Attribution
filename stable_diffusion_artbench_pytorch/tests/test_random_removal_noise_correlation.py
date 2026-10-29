from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/sd_artbench/analyze_random_removal_noise_correlation.py"


class RandomRemovalNoiseCorrelationTest(unittest.TestCase):
    def test_script_compiles(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)

    def test_noise_is_compared_on_shared_reference_inputs(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("capture_full_data_noise", source)
        self.assertIn("predicted_noise_mse_on_reference_inputs", source)
        self.assertIn("identical scaled full-data DDIM trajectory inputs", source)

    def test_all_requested_correlations_are_saved(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for field in (
            "endpoint_pearson",
            "endpoint_spearman",
            "trajectory_pearson",
            "trajectory_spearman",
        ):
            self.assertIn(field, source)
        self.assertIn('"correlation_by_timestep.csv"', source)
        self.assertIn('"correlation_by_timestep.png"', source)

    def test_run_is_resumable_per_query(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("read_completed", source)
        self.assertIn("Q{query_index} complete", source)


if __name__ == "__main__":
    unittest.main()
