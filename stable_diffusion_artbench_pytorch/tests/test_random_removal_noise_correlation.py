from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/sd_artbench/analyze_random_removal_noise_correlation.py"


class RandomRemovalNoiseCorrelationTest(unittest.TestCase):
    def test_script_compiles(self) -> None:
        ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def test_experiment_is_query_by_model_cross_product(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"--model-indices"', source)
        self.assertIn("for model in args.model_indices", source)
        self.assertIn('"random_model_index"', source)
        self.assertIn("len(records) * len(args.model_indices)", source)

    def test_noise_uses_shared_reference_inputs(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("capture_full_data_noise", source)
        self.assertIn("predicted_noise_mse_on_reference_inputs", source)
        self.assertIn("identical scaled full-data DDIM trajectory inputs", source)

    def test_correlations_are_within_query_across_models(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("compute_correlation_rows", source)
        self.assertIn("within each evaluation query across random models", source)

    def test_resumes_per_query_model_pair(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("completed_pairs", source)
        self.assertIn("cross_model_per_query_timestep.csv", source)

    def test_cached_csv_sort_converts_strings_to_ints(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('int(row["evaluation_query_index"])', source)
        self.assertIn('int(row["random_model_index"])', source)
        self.assertIn('int(row["step"])', source)


if __name__ == "__main__":
    unittest.main()
