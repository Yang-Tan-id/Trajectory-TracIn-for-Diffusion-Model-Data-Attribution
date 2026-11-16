from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "diffusion_jax_refined"
    / "celeba"
    / "script"
    / "analyze_das_top_bottom1000_endpoint_perceptual.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("das_top_bottom", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CelebaDASTopBottomPerceptualTest(unittest.TestCase):
    def test_select_extremes_uses_largest_and_smallest_scores(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = np.arange(10_000, dtype=np.float32)
            np.save(root / "scores.npy", scores)
            np.save(root / "score_indices.npy", np.arange(10_000, dtype=np.int64))
            top, bottom = module.select_extremes(root, 1000)
        self.assertEqual(set(top.tolist()), set(range(9000, 10_000)))
        self.assertEqual(set(bottom.tolist()), set(range(1000)))
        self.assertFalse(set(top.tolist()) & set(bottom.tolist()))

    def test_contract_is_ema_lambda200_and_reports_both_groups(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn('default="das_50x1_mc1_full10k_ema"', text)
        self.assertIn('default=200.0', text)
        self.assertIn('(query_id, "top")', text)
        self.assertIn('(query_id, "bottom")', text)
        self.assertIn('output_dir / "per_query.csv"', text)
        self.assertIn('output_dir / "summary.csv"', text)
        self.assertIn('output_dir / "report.txt"', text)


if __name__ == "__main__":
    unittest.main()
