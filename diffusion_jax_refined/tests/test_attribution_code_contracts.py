from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy_jax"


class TestAttributionCodeContracts(unittest.TestCase):
    def test_das_current_scores_are_squared_magnitude(self):
        text = (LEGACY / "das" / "algorithm.py").read_text()
        self.assertIn("np.square(raw)", text)
        self.assertIn("score_i = raw_i * raw_i", text)
        self.assertIn('np.save(os.path.join(out_dir, "scores.npy"), scores)', text)

    def test_traj_tracin_query_objective_is_noise_squared_deviation(self):
        text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn("trajectory_noise_squared_deviation", text)
        self.assertIn("eps_theta(x_ref_k,k)-eps_theta_ref(x_ref_k,k)", text)
        self.assertIn('np.save(os.path.join(out_dir, "score_indices.npy")', text)

    def test_cifar_jax_training_uses_cosine_warmup_lr_by_default(self):
        text = (LEGACY / "DM__training_CIFAR10_pixel.py").read_text()
        self.assertIn("learning_rate: float = 1e-4", text)
        self.assertIn('lr_schedule: str = "cosine_warmup"', text)
        self.assertIn("lr_warmup_ratio: float = 0.1", text)
        self.assertIn("optax.warmup_cosine_decay_schedule", text)
        self.assertIn("learning_rate=lr_schedule", text)
        self.assertIn('"train/lr": lr_val', text)
        self.assertIn("total_steps: Optional[int] = None", text)
        self.assertIn("if total_steps is None:", text)

    def test_traj_tracin_uses_checkpoint_lr_weights(self):
        text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn("tracin_checkpoint_lr_weight", text)
        self.assertIn('tracin_lr_schedule: str = "cosine_warmup"', text)
        self.assertIn('\"lr_schedule\": \"tracin_lr_schedule\"', text)
        self.assertIn("stage_term_weights.append(ckpt_lr_weight / float(max(1, len(t_seq))))", text)
        self.assertIn("snap_weight = ckpt_lr_weight / float(max(1, len(t_seq)))", text)

    def test_traj_tracin_score_combiner_rejects_mismatched_term_weights(self):
        text = (ROOT / "common" / "stage_artifact_runner.py").read_text()
        self.assertIn("train/query term_weights differ", text)
        self.assertIn("np.allclose(train_weights, weights", text)

    def test_traj_tracin_has_normalized_eps_deviation_objectives(self):
        text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn('query_objective: str = "trajectory_noise_squared_deviation"', text)
        self.assertIn('"eps_deviation_l1_mean"', text)
        self.assertIn("jnp.mean(jnp.abs(diff))", text)
        self.assertIn('"eps_deviation_l2_sq_mean"', text)
        self.assertIn("jnp.mean(diff ** 2)", text)
        self.assertIn('"trajectory_noise_squared_deviation_normalized"', text)

    def test_traj_tracin_can_write_paired_query_normalized_scores(self):
        text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn("save_query_normalized_scores: bool = False", text)
        self.assertIn("tree_l2_normalize(query_grad, query_normalize_eps)", text)
        self.assertIn("query_normalized_out_dir(cfg.out_dir)", text)
        self.assertIn('"score_variant"', text)
        self.assertIn('"query_gradient_l2_normalized"', text)
        self.assertIn('"train_gradient": "none"', text)

    def test_nondefault_traj_objective_gets_distinct_score_folder(self):
        text = (ROOT / "common" / "algorithm_runner.py").read_text()
        self.assertIn('if algorithm == "traj_tracin"', text)
        self.assertIn('output_algorithm = f"{algorithm}_{_safe_tag(objective)}"', text)

    def test_traj_tracin_normalized_eval_uses_range_inputs(self):
        cifar_eval = (ROOT / "cifar2" / "lds" / "CONFIG.py").read_text()
        common_eval = (ROOT / "common" / "lds_model_eval.py").read_text()
        self.assertIn('algorithm.startswith("traj_tracin") and ranges', cifar_eval)
        self.assertIn('args.algorithm.startswith("traj_tracin") and ranges', common_eval)

    def test_stage_runner_does_not_call_full_attribution_entrypoint(self):
        text = (ROOT / "common" / "stage_runner.py").read_text()
        self.assertIn('"train_datapoint_gradient"', text)
        self.assertIn('"query_gradient"', text)
        self.assertIn('"score"', text)
        self.assertIn('f"seed_{train_seed}_train_gradient"', text)
        self.assertIn('f"seed_{sample_seed:06d}_query_gradient"', text)
        self.assertIn("def canonical_train_model_mode", text)
        self.assertIn('"prompted_multi"', text)
        self.assertIn('return "prompted_solo"', text)
        self.assertIn('"unprompted_multi"', text)
        self.assertIn('return "unprompted_solo"', text)
        self.assertIn('"ATTRIBUTION_SCORE_MODEL_MODE"', text)
        self.assertIn('"ATTRIBUTION_SAMPLE_MODEL_MODE"', text)
        self.assertIn('"SAMPLE_MODEL_MODE"', text)
        self.assertNotIn("run_algorithm_config", text)
        self.assertNotIn("run_attribution", text)

    def test_all_score_stages_are_pure_artifact_combiners(self):
        for dataset in ("cifar2", "cifar10", "artbench"):
            for algorithm in ("das", "dtrak", "end_tracin", "traj_tracin", "journey_trak"):
                with self.subTest(dataset=dataset, algorithm=algorithm):
                    text = (ROOT / dataset / "data_attribution" / algorithm / "03_score.py").read_text()
                    self.assertIn("run_score_combination_stage", text)
                    self.assertNotIn("run_algorithm_config", text)
                    self.assertNotIn("run_attribution", text)

    def test_score_artifact_combiner_dot_scores(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not installed for this Python")
        np = __import__("numpy")
        from common.stage_artifact_runner import _combine_dot_scores

        train = {
            "train_features": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "score_indices": np.asarray([10, 11], dtype=np.int64),
        }
        query = {"query_feature": np.asarray([5.0, 6.0], dtype=np.float32)}
        scores = _combine_dot_scores(train, query, train_path=Path("train.npz"), query_path=Path("query.npz"))
        np.testing.assert_allclose(scores, np.asarray([17.0, 39.0]))

    def test_score_artifact_combiner_das_uses_residual_and_inverse_gram(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not installed for this Python")
        np = __import__("numpy")
        from common.stage_artifact_runner import _combine_das_scores

        train = {
            "train_features": np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            "gram_inverse": np.asarray([[0.2, 0.0], [0.0, 0.05]], dtype=np.float32),
        }
        query = {
            "query_feature": np.asarray([5.0, 6.0], dtype=np.float32),
            "residuals": np.asarray([3.0, 4.0], dtype=np.float32),
        }
        scores = _combine_das_scores(train, query, train_path=Path("train.npz"), query_path=Path("query.npz"))
        np.testing.assert_allclose(scores, np.asarray([14.0625, 9.0]))

    def test_score_artifact_combiner_dtrak_uses_gram_solve(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not installed for this Python")
        np = __import__("numpy")
        from common.stage_artifact_runner import _combine_dtrak_scores

        train = {
            "train_features": np.asarray([[[1.0, 0.0], [0.0, 2.0]]], dtype=np.float32),
            "gram": np.asarray([[[2.0, 0.0], [0.0, 4.0]]], dtype=np.float32),
        }
        query = {"query_features": np.asarray([[6.0, 8.0]], dtype=np.float32)}
        scores = _combine_dtrak_scores(train, query, train_path=Path("train.npz"), query_path=Path("query.npz"))
        np.testing.assert_allclose(scores, np.asarray([3.0, 4.0]))

    def test_dtrak_stage_producer_is_wired_to_real_legacy_stage_mode(self):
        text = (ROOT / "common" / "stage_artifact_producer.py").read_text()
        self.assertIn('"DTRAK_STAGE_MODE": "train"', text)
        self.assertIn('"DTRAK_STAGE_MODE": "query"', text)
        self.assertIn("run_algorithm_config(config_path)", text)
        dtrak_text = (LEGACY / "dtrak" / "algorithm.py").read_text()
        self.assertIn('stage_mode = os.environ.get("DTRAK_STAGE_MODE"', dtrak_text)
        self.assertIn("np.savez_compressed(", dtrak_text)
        self.assertIn("train_features=np.stack(stage_train_features", dtrak_text)
        self.assertIn("query_features=np.stack(stage_query_features", dtrak_text)

    def test_original_algorithm_runner_uses_unique_output_dirs_for_ranges_and_unprompted(self):
        text = (ROOT / "common" / "algorithm_runner.py").read_text()
        self.assertIn("def _range_suffix_from_env", text)
        self.assertIn('os.environ.get("ATTRIBUTION_RANGES") or os.environ.get("SCORE_INDEX_RANGES")', text)
        self.assertIn('output_algorithm = f"{output_algorithm}_unprompted"', text)
        self.assertIn('output_algorithm = f"{output_algorithm}_{range_suffix}"', text)
        self.assertIn("ATTRIBUTION_RANGES/SCORE_INDEX_RANGES must look like", text)

    def test_strict_stage_producer_does_not_use_score_artifact_fallback(self):
        text = (ROOT / "common" / "stage_artifact_producer.py").read_text()
        self.assertNotIn("score_artifact_dir", text)
        self.assertNotIn("ATTRIBUTION_OUT_DIR", text)
        self.assertIn('"DAS_STAGE_MODE"', text)
        self.assertIn('"END_TRACIN_STAGE_MODE"', text)
        self.assertIn('"TRAJ_TRACIN_STAGE_MODE"', text)

    def test_das_sherman_morrison_denominator_is_explicit_option(self):
        text = (LEGACY / "das" / "algorithm.py").read_text()
        self.assertIn("use_sherman_morrison_denominator: bool = False", text)
        self.assertIn("raw /= denominator", text)
        self.assertIn("denom_i = 1.0 - leverage_i", text)
        self.assertIn("[batched] solving damped projected Gram", text)
        self.assertIn("[batched] applying Sherman-Morrison denominator", text)
        self.assertIn("def make_spd_solver", text)
        self.assertIn("cho_factor(matrix", text)
        self.assertIn("solve_h_proj = make_spd_solver(H_proj)", text)
        self.assertIn("DAS denominator lambda=", text)
        self.assertIn("scores_by_damping", text)
        self.assertIn('out_dir = os.path.join(save_root, f"lambda_{damping_output_tag(damping_value)}")', text)
        batched_block = text.split("def compute_batched_das_term", 1)[1].split("# ============================================================\n# Main", 1)[0]
        self.assertNotIn("np.linalg.solve(H_proj", batched_block)
        cifar2_text = (ROOT / "cifar2" / "dataset_config.py").read_text()
        self.assertIn("DAS_SHERMAN_MORRISON_DENOMINATOR", cifar2_text)

    def test_das_damping_defaults_to_two_and_is_env_tunable(self):
        text = (LEGACY / "das" / "algorithm.py").read_text()
        self.assertIn("damping: float = 2.0", text)
        for dataset in ("cifar2", "cifar10", "artbench"):
            with self.subTest(dataset=dataset):
                dataset_text = (ROOT / dataset / "dataset_config.py").read_text()
                self.assertIn('"damping": float(os.environ.get("DAS_DAMPING", "2"))', dataset_text)

    def test_das_projection_dimension_is_env_tunable_for_smoke_runs(self):
        for dataset in ("cifar2", "cifar10", "artbench"):
            with self.subTest(dataset=dataset):
                dataset_text = (ROOT / dataset / "dataset_config.py").read_text()
                self.assertIn('"proj_dim": int(os.environ.get("DAS_PROJ_DIM", "4096"))', dataset_text)
                self.assertIn("def _parse_int_list_env", dataset_text)
                self.assertIn('"timesteps": _parse_int_list_env("DAS_TIMESTEPS"', dataset_text)
                self.assertIn('"num_mc_noise": int(os.environ.get("DAS_NUM_MC_NOISE", "10"))', dataset_text)

    def test_das_damping_sweep_values_are_pinned(self):
        expected_values = (
            "0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0", "20.0", "50.0",
        )
        excluded_values = (
            "0.01", "0.02", "0.05", "100.0", "200.0", "500.0", "1000.0",
            "2000.0", "5000.0", "10000.0", "20000.0", "50000.0",
        )
        text = (LEGACY / "das" / "algorithm.py").read_text()
        self.assertIn("damping_sweep_values: Tuple[float, ...]", text)
        sweep_block = text.split("damping_sweep_values: Tuple[float, ...]", 1)[1].split("proj_dim: int", 1)[0]
        for value in expected_values:
            self.assertIn(value, sweep_block)
        for value in excluded_values:
            self.assertNotIn(value, sweep_block)
        for dataset in ("cifar2", "cifar10", "artbench"):
            with self.subTest(dataset=dataset):
                dataset_text = (ROOT / dataset / "dataset_config.py").read_text()
                self.assertIn("DAS_DAMPING_SWEEP_VALUES", dataset_text)
                self.assertIn('"damping_sweep_values": DAS_DAMPING_SWEEP_VALUES', dataset_text)
                dataset_sweep_block = dataset_text.split("DAS_DAMPING_SWEEP_VALUES = _parse_float_list_env", 1)[1].split("TRAIN_SEED =", 1)[0]
                for value in expected_values:
                    self.assertIn(value, dataset_sweep_block)
                for value in excluded_values:
                    self.assertNotIn(value, dataset_sweep_block)

    def test_end_tracin_uses_endpoint_anchored_loss_with_mc_samples(self):
        text = (LEGACY / "end_tracin" / "algorithm.py").read_text()
        self.assertIn("endpoint_anchored_loss_mc", text)
        self.assertIn("endpoint_mc_samples: int = 10", text)
        self.assertIn("train_mc_samples: int = 10", text)
        self.assertIn("sc = eta_k * tree_vdot(g_end, g_tr)", text)

    def test_cifar2_algorithm_sample_counts_are_pinned(self):
        text = (ROOT / "cifar2" / "dataset_config.py").read_text()
        self.assertIn('os.environ.get("END_TRACIN_ENDPOINT_MC_SAMPLES", "10")', text)
        self.assertIn('os.environ.get("END_TRACIN_TRAIN_MC_SAMPLES", "10")', text)
        self.assertIn('"train_mc_samples": 10', text)
        self.assertIn('"num_mc_noise": int(os.environ.get("DAS_NUM_MC_NOISE", "10"))', text)
        self.assertIn('os.environ.get("DTRAK_QUERY_EXPECTATION_SAMPLES", "10")', text)
        self.assertIn('os.environ.get("DTRAK_TRAIN_EXPECTATION_SAMPLES", "10")', text)

    def test_batch_sizes_are_tunable_for_gpu_utilization(self):
        for dataset in ("cifar2", "cifar10", "artbench"):
            with self.subTest(dataset=dataset):
                text = (ROOT / dataset / "dataset_config.py").read_text()
                self.assertIn("TRAJ_SNAPSHOT_CHUNK_SIZE", text)
                self.assertIn("DTRAK_BATCH_SIZE", text)
                self.assertIn("JOURNEY_BATCH_SIZE", text)

    def test_attribution_progress_logs_prefer_single_tqdm_bar(self):
        for algorithm in ("das", "dtrak", "traj_tracin"):
            with self.subTest(algorithm=algorithm):
                text = (LEGACY / algorithm / "algorithm.py").read_text()
                self.assertIn("ATTRIBUTION_TQDM_MININTERVAL", text)
                self.assertIn("ATTRIBUTION_TQDM_LEAVE", text)
                if algorithm in ("das", "dtrak"):
                    self.assertIn("(not cfg.use_tqdm) and should_print_progress", text)
                else:
                    self.assertIn("if (not cfg.use_tqdm) and (", text)

        end_text = (LEGACY / "end_tracin" / "algorithm.py").read_text()
        self.assertIn("ATTRIBUTION_TQDM_MININTERVAL", end_text)
        vista_text = (ROOT / "cifar2" / "vista_original" / "02_attribution_chunk_vista.sh").read_text()
        self.assertIn("PYTHONUNBUFFERED=1", vista_text)
        self.assertIn("ATTRIBUTION_TQDM_MININTERVAL", vista_text)

    def test_dtrak_train_features_are_batched_with_vmap(self):
        text = (LEGACY / "dtrak" / "algorithm.py").read_text()
        self.assertIn("def make_train_phi_batch_fn", text)
        self.assertIn("jax.vmap(train_phi_fn", text)
        self.assertIn("train_phi_batch_fn(params_k, x_batch, cond_batch, rng_batch)", text)
        self.assertIn("phi_cache = np.empty((M, d), dtype=np.float32)", text)
        self.assertIn("phi_cache[start:start + len(batch)] = Phi", text)
        self.assertIn("Phi = phi_cache[start:start + len(batch)]", text)
        self.assertEqual(text.count("train_phi_batch_fn(params_k, x_batch, cond_batch, rng_batch)"), 1)

    def test_algorithms_expose_three_stage_facades(self):
        expected_exports = {
            "das": ("compute_train_gradient_features", "compute_query_gradient_features", "compute_scores"),
            "dtrak": ("compute_train_gradient_features", "compute_query_gradient_features", "compute_scores"),
            "end_tracin": ("compute_train_gradients", "compute_query_gradient", "compute_scores"),
            "traj_tracin": ("compute_train_gradients", "compute_query_gradients", "compute_scores"),
            "journey_trak": ("compute_train_gradient_features", "compute_query_gradient_features", "compute_scores"),
        }
        for algorithm, names in expected_exports.items():
            with self.subTest(algorithm=algorithm):
                text = (LEGACY / algorithm / "stages.py").read_text()
                for name in names:
                    self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
