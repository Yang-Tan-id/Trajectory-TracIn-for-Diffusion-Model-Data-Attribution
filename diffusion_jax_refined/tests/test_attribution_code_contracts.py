from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy_jax"


class TestAttributionCodeContracts(unittest.TestCase):
    def test_das_current_scores_are_squared_magnitude(self):
        text = (LEGACY / "DM_dataAttribution_algo_end_das.py").read_text()
        self.assertIn("return np.square(raw)", text)
        self.assertIn("score_i = raw_i * raw_i", text)
        self.assertIn('np.save(os.path.join(cfg.out_dir, "scores.npy"), scores)', text)

    def test_traj_tracin_query_objective_is_noise_squared_deviation(self):
        text = (LEGACY / "DM_dataAttribution_algo_traj_tracin.py").read_text()
        self.assertIn("trajectory_noise_squared_deviation", text)
        self.assertIn("eps_theta(x_ref_k,k)-eps_theta_ref(x_ref_k,k)", text)
        self.assertIn('np.save(os.path.join(cfg.out_dir, "score_indices.npy")', text)

    def test_traj_tracin_has_normalized_eps_deviation_objectives(self):
        text = (LEGACY / "DM_dataAttribution_algo_traj_tracin.py").read_text()
        self.assertIn('query_objective: str = "trajectory_noise_squared_deviation"', text)
        self.assertIn('"eps_deviation_l1_mean"', text)
        self.assertIn("jnp.mean(jnp.abs(diff))", text)
        self.assertIn('"eps_deviation_l2_sq_mean"', text)
        self.assertIn("jnp.mean(diff ** 2)", text)
        self.assertIn('"trajectory_noise_squared_deviation_normalized"', text)

    def test_nondefault_traj_objective_gets_distinct_score_folder(self):
        text = (ROOT / "common" / "algorithm_runner.py").read_text()
        self.assertIn('if algorithm == "traj_tracin"', text)
        self.assertIn('output_algorithm = f"{algorithm}_{_safe_tag(objective)}"', text)

    def test_das_sherman_morrison_denominator_is_explicit_option(self):
        text = (LEGACY / "DM_dataAttribution_algo_end_das.py").read_text()
        self.assertIn("use_sherman_morrison_denominator: bool = False", text)
        self.assertIn("raw /= denominator", text)
        self.assertIn("denom_i = 1.0 - leverage_i", text)
        cifar2_text = (ROOT / "cifar2" / "dataset_config.py").read_text()
        self.assertIn("DAS_SHERMAN_MORRISON_DENOMINATOR", cifar2_text)

    def test_end_tracin_uses_endpoint_anchored_loss_with_mc_samples(self):
        text = (LEGACY / "DM_dataAttribution_algo_end_tracin.py").read_text()
        self.assertIn("endpoint_anchored_loss_mc", text)
        self.assertIn("endpoint_mc_samples: int = 8", text)
        self.assertIn("train_mc_samples: int = 8", text)
        self.assertIn("sc = eta_k * tree_vdot(g_end, g_tr)", text)

    def test_cifar2_algorithm_sample_counts_are_pinned(self):
        text = (ROOT / "cifar2" / "dataset_config.py").read_text()
        self.assertIn('"endpoint_mc_samples": 8', text)
        self.assertIn('"train_mc_samples": 8', text)
        self.assertIn('"num_mc_noise": 8', text)
        self.assertIn('"query_expectation_samples": 8', text)
        self.assertIn('"train_expectation_samples": 8', text)


if __name__ == "__main__":
    unittest.main()
