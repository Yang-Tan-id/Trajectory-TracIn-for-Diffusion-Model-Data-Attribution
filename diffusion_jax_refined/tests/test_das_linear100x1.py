import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DasLinear100x1Tests(unittest.TestCase):
    def test_linear_and_squared_term_reductions_differ(self):
        raw_terms = np.asarray([[2.0, -3.0], [-1.0, 1.0]])
        linear = raw_terms.mean(axis=0)
        squared = np.square(raw_terms).mean(axis=0)
        np.testing.assert_allclose(linear, [0.5, -1.0])
        np.testing.assert_allclose(squared, [2.5, 5.0])

    def test_artifact_combiner_preserves_sign_in_linear_mode(self):
        from diffusion_jax_refined.common.stage_artifact_runner import _combine_das_scores

        train_payload = {
            "train_features": np.asarray([[[2.0], [-3.0]], [[-1.0], [1.0]]]),
            "residuals": np.ones((2, 2)),
        }
        query_payload = {"query_features": np.ones((2, 1))}
        env = {
            "DAS_SHERMAN_MORRISON_DENOMINATOR": "0",
            "DAS_SCORE_TQDM": "0",
            "DAS_SCORE_BACKEND": "numpy",
        }
        with patch.dict(os.environ, env | {"DAS_SCORE_CONTRACTION": "linear"}):
            linear = _combine_das_scores(
                train_payload,
                query_payload,
                train_path=Path("train.npz"),
                query_path=Path("query.npz"),
            )
        with patch.dict(os.environ, env | {"DAS_SCORE_CONTRACTION": "squared"}):
            squared = _combine_das_scores(
                train_payload,
                query_payload,
                train_path=Path("train.npz"),
                query_path=Path("query.npz"),
            )
        np.testing.assert_allclose(linear, [0.5, -1.0])
        np.testing.assert_allclose(squared, [2.5, 5.0])

    def test_score_runner_supports_isolated_linear_output(self):
        driver = (ROOT / "3dshapes" / "script" / "run_das_queries_and_scores.py").read_text()
        scorer = (ROOT / "common" / "stage_artifact_runner.py").read_text()
        self.assertIn('choices=("squared", "linear")', driver)
        self.assertIn("--score-output-namespace", driver)
        self.assertIn('DAS_SCORE_CONTRACTION=args.score_contraction', driver)
        self.assertIn('raw if score_contraction == "linear"', scorer)
        self.assertIn('raw.T if score_contraction == "linear"', scorer)

    def test_pipeline_reuses_artifacts_and_uses_positive_lds_sign(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_das_linear100x1_pipeline_rtx_small.sh"
        ).read_text()
        self.assertIn("--skip-query-gradient", launcher)
        self.assertIn("--score-output-namespace linear100x1", launcher)
        self.assertIn("--score-contraction linear", launcher)
        self.assertIn("--artifact-namespace linear100x1", launcher)
        self.assertIn("--prediction-sign 1", launcher)
        self.assertIn("--prediction-sign p1", launcher)


if __name__ == "__main__":
    unittest.main()
