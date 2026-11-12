from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CELEBA = ROOT / "diffusion_jax_refined" / "celeba"


class CelebaDAS50x1PipelineTest(unittest.TestCase):
    def test_config_is_exactly_50_by_1_and_fixed_5k(self) -> None:
        path = CELEBA / "dataset_config_das_attr40.py"
        spec = importlib.util.spec_from_file_location("celeba_das_test_config", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.path.insert(0, str(CELEBA))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(CELEBA))
        config = module.ATTRIBUTION_CONFIGS["das"]
        self.assertEqual(config["timesteps"], tuple(range(19, 1000, 20)))
        self.assertEqual(len(config["timesteps"]), 50)
        self.assertEqual(config["num_mc_noise"], 1)
        self.assertEqual(config["max_train_points"], 5000)
        self.assertTrue(config["score_indices_path"].endswith("attribution_5k_indices.npy"))
        self.assertFalse(config["use_sherman_morrison_denominator"])

    def test_full10k_runner_disables_fixed_index_file(self) -> None:
        text = (
            CELEBA / "script" / "run_das_50x1_lambda_sweep_full10k_4gpu.sh"
        ).read_text()
        self.assertIn('DAS_NAMESPACE="${DAS_NAMESPACE:-das_50x1_mc1_full10k}"', text)
        self.assertIn('DAS_QUERY_ARTIFACT_NAMESPACE="${DAS_QUERY_ARTIFACT_NAMESPACE:-das_50x1_mc1}"', text)
        self.assertIn("export DAS_MAX_TRAIN_POINTS=10000", text)
        self.assertIn('export DAS_SCORE_INDICES_PATH=""', text)

    def test_runner_validates_dynamic_candidate_count(self) -> None:
        text = (CELEBA / "script" / "run_das_50x1_lambda_sweep_4gpu.sh").read_text()
        self.assertIn('num_points="${DAS_MAX_TRAIN_POINTS:-5000}"', text)
        self.assertIn("(50, expected_points, 4096)", text)
        self.assertIn("np.arange(10000)", text)

    def test_runner_has_four_disjoint_timestamp_shards(self) -> None:
        text = (CELEBA / "script" / "run_das_50x1_lambda_sweep_4gpu.sh").read_text()
        values = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('"') and stripped.endswith('"'):
                body = stripped.strip('"')
                if body and all(part.isdigit() for part in body.split(",")):
                    values.extend(int(part) for part in body.split(","))
        self.assertEqual(values, list(range(19, 1000, 20)))
        self.assertEqual(len(values), len(set(values)))
        self.assertIn("DAS_SHERMAN_MORRISON_DENOMINATOR=0", text)
        self.assertIn('DAS_GRAD_BATCH_SIZE="${DAS_GRAD_BATCH_SIZE:-8}"', text)

    def test_das_engine_accepts_an_explicit_index_file(self) -> None:
        text = (
            ROOT / "diffusion_jax_refined" / "legacy_jax" / "das" / "algorithm.py"
        ).read_text()
        self.assertIn("score_indices_path: Optional[str] = None", text)
        self.assertIn("np.load(path)", text)
        self.assertIn("DAS score index file contains duplicates", text)

    def test_staged_train_path_uses_true_vmap_microbatches(self) -> None:
        text = (
            ROOT / "diffusion_jax_refined" / "legacy_jax" / "das" / "algorithm.py"
        ).read_text()
        self.assertIn("def compute_batched_das_train_artifact_term(", text)
        self.assertIn('"[stage-train] using vmapped exact per-example gradients | "', text)
        self.assertIn("jax.vmap(phi_fn", text)
        self.assertIn(
            "np.array(gram_device, dtype=np.float32, copy=True)",
            text,
        )

    def test_query_stage_batches_queries_timestamp_major(self) -> None:
        engine = (
            ROOT / "diffusion_jax_refined" / "legacy_jax" / "das" / "algorithm.py"
        ).read_text()
        runner = (
            CELEBA / "script" / "run_das_50x1_queries_and_scores.py"
        ).read_text()
        self.assertIn('DAS_QUERY_BATCH_JOBS', engine)
        self.assertIn('stage_query_features_by_job', engine)
        self.assertIn('jax.vmap(', engine)
        self.assertIn('"DAS_QUERY_BATCH_JOBS": json.dumps(jobs)', runner)

    def test_das_pint_comparison_uses_all_aligned_queries(self) -> None:
        text = (
            CELEBA / "script" / "compare_das_pint_50x1_lds.py"
        ).read_text()
        self.assertIn('parser.add_argument("--query-ids", default="0-71")', text)
        self.assertIn('method_configs("celeba")[:2]', text)
        self.assertIn('best_lambda[target] = max(candidates)[1]', text)
        self.assertIn('sign = -1.0', text)


if __name__ == "__main__":
    unittest.main()

