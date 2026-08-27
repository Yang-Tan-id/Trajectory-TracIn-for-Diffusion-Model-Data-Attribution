from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile


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

    def test_projected_traj_tracin_score_sweep_can_ignore_lr_weights(self):
        text = (ROOT / "common" / "projected_traj_tracin_score_sweep.py").read_text()
        self.assertIn("checkpoint_uniform_term_weights", text)
        self.assertIn("--term-weighting", text)
        self.assertIn("uniform_checkpoint", text)
        self.assertIn("term_weighting", text)

        h100_text = (
            ROOT / "cifar2" / "tacc" / "h100" / "projected_traj_tracin_score_sweep.sh"
        ).read_text()
        self.assertIn("TRAJ_TRACIN_TERM_WEIGHTING", h100_text)
        self.assertIn("traj_tracin_projected_${TRAJ_TRACIN_TERM_WEIGHTING}", h100_text)
        self.assertIn('--term-weighting "${TRAJ_TRACIN_TERM_WEIGHTING}"', h100_text)

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

    def test_traj_tracin_has_next_checkpoint_noise_target(self):
        text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn('"trajectory_next_checkpoint_noise_mse"', text)
        self.assertIn("eps_theta_c_plus_1", text)
        self.assertIn("query_objective_uses_next_checkpoint", text)
        self.assertIn("no next-checkpoint query target", text)
        self.assertIn('"target_checkpoint"', text)

    def test_traj_tracin_can_write_paired_query_normalized_scores(self):
        text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn("save_query_normalized_scores: bool = False", text)
        self.assertIn("tree_l2_normalize(query_grad, query_normalize_eps)", text)
        self.assertIn("query_normalized_out_dir(cfg.out_dir)", text)
        self.assertIn('"score_variant"', text)
        self.assertIn('"query_gradient_l2_normalized"', text)
        self.assertIn('"train_gradient": "none"', text)

    def test_cifar2_traj_tracin_saved_trajectory_is_env_tunable_for_projected_cache(self):
        text = (ROOT / "cifar2" / "dataset_config.py").read_text()
        self.assertIn("TRAJ_USE_SAVED_TRAJECTORY", text)
        self.assertIn('"use_saved_trajectory": os.environ.get("TRAJ_USE_SAVED_TRAJECTORY", "1")', text)

        traj_text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn("def ensure_dir(path: str) -> None:", traj_text)
        self.assertIn("def save_npz_compressed_atomic(path: str, **arrays) -> None:", traj_text)
        self.assertIn("os.makedirs(path, exist_ok=True)", traj_text)
        self.assertIn('tmp_path = f"{path}.tmp.npz"', traj_text)
        self.assertIn("os.replace(tmp_path, path)", traj_text)
        self.assertIn("ensure_dir(os.path.dirname(path))", traj_text)
        self.assertIn('stage_part_dir = f"{stage_artifact_path}.parts"', traj_text)
        self.assertIn('f"ckpt_{ckpt_i:04d}.npz"', traj_text)
        self.assertIn("skip existing checkpoint part", traj_text)
        self.assertIn("[stage:train] saved checkpoint part", traj_text)
        self.assertIn("TrajTracIn train checkpoint parts are incomplete", traj_text)
        self.assertIn("np.concatenate(train_features_parts, axis=0)", traj_text)

        projected_text = (
            ROOT / "cifar2" / "tacc" / "h100" / "projected_traj_tracin_score_sweep.sh"
        ).read_text()
        self.assertIn("TRAJ_USE_SAVED_TRAJECTORY=0", projected_text)
        self.assertIn('run_original_config_with_stage train "${artifact_path}"', projected_text)

    def test_traj_tracin_has_query_cached_stream_score_stage(self):
        traj_text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn('"score_stream"', traj_text)
        self.assertIn("load_stream_query_bank", traj_text)
        self.assertIn("TRAJ_TRACIN_STREAM_QUERY_ARTIFACTS", traj_text)
        self.assertIn("TRAJ_TRACIN_STREAM_PROJ_DIMS", traj_text)
        self.assertIn("scores_raw = np.zeros((num_dims, num_queries, len(picked))", traj_text)
        self.assertIn("TRAJ_TRACIN_STREAM_SAVE_TERM_SCORE_VARIANTS", traj_text)
        self.assertIn("scores_by_term_train_l2_normalized", traj_text)
        self.assertIn("phi = train_phi_batch(params, x_batch, cond_batch, rngs, t_scalar)", traj_text)
        self.assertIn("raw = phi_dim @ q_raw_by_dim[dim_i].T", traj_text)
        self.assertIn("scores_query_train_l2_normalized", traj_text)

        h100_script = ROOT / "cifar2" / "tacc" / "h100" / "projected_traj_tracin_stream_score_array_h100.sh"
        self.assertTrue(h100_script.is_file())
        script_text = h100_script.read_text()
        self.assertIn("#SBATCH -J cifar2-proj-traj-stream", script_text)
        self.assertIn("SLURM_SUBMIT_DIR", script_text)
        self.assertIn("SCRIPT_DIR_CANDIDATE", script_text)
        self.assertIn("Could not locate projected_traj_tracin_score_sweep.sh", script_text)
        self.assertIn("STREAM_PROJ_DIMS", script_text)
        self.assertIn("STREAM_QUERY_FILTERS", script_text)
        self.assertIn("STREAM_SAVE_TERM_SCORE_VARIANTS", script_text)
        self.assertIn("stream_term_scores", script_text)
        self.assertIn("TRAJ_TRACIN_STAGE_MODE=score_stream", script_text)
        self.assertIn("TRAJ_TRACIN_STREAM_QUERY_ARTIFACTS", script_text)
        self.assertIn("merge_stream_score_shards.py", script_text)

    def test_traj_tracin_can_save_full_dim_term_scores(self):
        traj_text = (LEGACY / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn("TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS", traj_text)
        self.assertIn("TRAJ_TRACIN_FULL_TERM_SCORE_ARTIFACT_PATH", traj_text)
        self.assertIn("scores_by_term_raw", traj_text)
        self.assertIn("full-dim per-term TrajTracIn score artifact", traj_text)
        self.assertIn("term_ckpt_indices", traj_text)
        self.assertIn("term_timesteps", traj_text)

        h100_script = ROOT / "cifar2" / "tacc" / "h100" / "full_traj_tracin_4query_term_scores_h100.sh"
        self.assertTrue(h100_script.is_file())
        script_text = h100_script.read_text()
        self.assertIn("#SBATCH -J cifar2-full-traj-term", script_text)
        self.assertIn("FULL_QUERY_SPECS", script_text)
        self.assertIn("FULL_SCORE_RANGES", script_text)
        self.assertIn("TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS", script_text)
        self.assertIn("full_dim_term_scores.npz", script_text)
        self.assertIn("run_original_attribution_config.py", script_text)

    def test_merge_stream_score_shards_orders_indices_and_preserves_dims(self):
        try:
            import numpy as np
        except Exception as exc:
            self.skipTest(f"numpy cannot be imported: {exc}")

        merge_script = ROOT / "common" / "merge_stream_score_shards.py"
        self.assertTrue(merge_script.is_file())
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            query_artifacts = np.asarray(["q0.npz", "q1.npz"])
            for shard_id, indices in enumerate((np.asarray([3, 4]), np.asarray([1, 2]))):
                base = np.full((2, 2, 2), float(shard_id), dtype=np.float64)
                np.savez_compressed(
                    tmp / f"shard_{shard_id}.npz",
                    scores_raw=base,
                    scores_query_l2_normalized=base + 10.0,
                    scores_train_l2_normalized=base + 20.0,
                    scores_query_train_l2_normalized=base + 30.0,
                    scores_by_term_train_l2_normalized=np.full((3, 2, 2, 2), float(shard_id), dtype=np.float32),
                    score_indices=indices,
                    query_artifacts=query_artifacts,
                    proj_dims=np.asarray([256, 512], dtype=np.int32),
                    term_ckpt_indices=np.asarray([0, 0, 1], dtype=np.int32),
                    term_timesteps=np.asarray([1, 2, 3], dtype=np.int32),
                    term_score_variants=np.asarray(["train_l2_normalized"]),
                )
            out = tmp / "merged.npz"
            subprocess.check_call(
                [
                    sys.executable,
                    str(merge_script),
                    "--output",
                    str(out),
                    str(tmp / "shard_0.npz"),
                    str(tmp / "shard_1.npz"),
                ]
            )
            with np.load(out, allow_pickle=False) as merged:
                np.testing.assert_array_equal(merged["score_indices"], np.asarray([1, 2, 3, 4]))
                np.testing.assert_array_equal(merged["proj_dims"], np.asarray([256, 512], dtype=np.int32))
                self.assertEqual(tuple(merged["scores_raw"].shape), (2, 2, 4))
                self.assertEqual(tuple(merged["scores_by_term_train_l2_normalized"].shape), (3, 2, 2, 4))
                np.testing.assert_allclose(merged["scores_raw"][:, :, :2], 1.0)
                np.testing.assert_allclose(merged["scores_raw"][:, :, 2:], 0.0)
                np.testing.assert_allclose(merged["scores_by_term_train_l2_normalized"][:, :, :, :2], 1.0)
                np.testing.assert_allclose(merged["scores_by_term_train_l2_normalized"][:, :, :, 2:], 0.0)
                np.testing.assert_array_equal(merged["term_ckpt_indices"], np.asarray([0, 0, 1], dtype=np.int32))

    def test_fast_lds_stream_score_eval_selects_query_and_rewrites_subset_dirs(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not installed for this Python")
        np = __import__("numpy")

        script_path = ROOT / "common" / "fast_lds_stream_score_eval.py"
        self.assertTrue(script_path.is_file())
        spec = importlib.util.spec_from_file_location("fast_lds_stream_score_eval", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.path.insert(0, str(ROOT))
        spec.loader.exec_module(module)

        query_artifacts = np.asarray(
            [
                "/cache/query_horse/initial_seed_0/shared_query/proj_4096/query_gradient_artifact.npz",
                "/cache/query_automobile/initial_seed_0/shared_query/proj_4096/query_gradient_artifact.npz",
            ]
        )
        self.assertEqual(module.select_query_index(query_artifacts, "query_horse/initial_seed_0"), 0)
        self.assertEqual(
            module.rewrite_path(
                "/work2/user/repo/result/lds_model/models/subset_0000",
                [("/work2/user/repo", "/tmp/local/repo")],
            ),
            "/tmp/local/repo/result/lds_model/models/subset_0000",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = Path(tmpdir) / "stream_scores_merged.npz"
            np.savez_compressed(
                merged_path,
                scores_raw=np.asarray([[[1.0, 2.0], [3.0, 4.0]]]),
                scores_query_l2_normalized=np.asarray([[[5.0, 6.0], [7.0, 8.0]]]),
                scores_train_l2_normalized=np.asarray([[[9.0, 10.0], [11.0, 12.0]]]),
                scores_query_train_l2_normalized=np.asarray([[[13.0, 14.0], [15.0, 16.0]]]),
                score_indices=np.asarray([20, 21], dtype=np.int64),
                query_artifacts=query_artifacts,
                proj_dims=np.asarray([1024], dtype=np.int32),
            )
            with np.load(merged_path, allow_pickle=True) as payload:
                score_map = module.score_map_from_stream(
                    payload,
                    query_index=1,
                    proj_dim=1024,
                    variant="query_train_l2_normalized",
                )
            self.assertEqual(score_map, {20: 15.0, 21: 16.0})

    def test_full_dim_term_weighted_eval_applies_ckpt_snapshot_weights(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not installed for this Python")
        np = __import__("numpy")

        script_path = ROOT / "common" / "full_dim_term_weighted_lds_eval.py"
        self.assertTrue(script_path.is_file())
        spec = importlib.util.spec_from_file_location("full_dim_term_weighted_lds_eval", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard = tmp / "full_dim_term_scores.npz"
            np.savez_compressed(
                shard,
                score_indices=np.asarray([10, 11], dtype=np.int64),
                term_ckpt_indices=np.asarray([0, 1, 1], dtype=np.int32),
                term_timesteps=np.asarray([999, 989, 979], dtype=np.int32),
                term_snapshot_positions=np.asarray([0, 500, 999], dtype=np.int32),
                scores_by_term_raw=np.asarray(
                    [
                        [1.0, 2.0],
                        [10.0, 20.0],
                        [100.0, 200.0],
                    ],
                    dtype=np.float32,
                ),
            )
            weights = np.asarray(
                [
                    [2.0, 1.0, 1.0],
                    [1.0, 3.0, 4.0],
                ],
                dtype=np.float64,
            )
            score_map, meta = module.score_map_from_term_artifacts(
                [shard],
                score_key="scores_by_term_raw",
                term_weight=weights,
            )
            self.assertEqual(meta["num_terms"], 3)
            self.assertEqual(score_map, {10: 432.0, 11: 864.0})

    def test_predicted_noise_change_weight_script_outputs_reusable_tables(self):
        text = (ROOT / "common" / "predicted_noise_change_weights.py").read_text()
        self.assertIn("eps_{theta_{c+1}}(x_s, t_s)", text)
        self.assertIn("eps_ref_mse_by_ckpt_snapshot", text)
        self.assertIn("delta_by_ckpt_snapshot", text)
        self.assertIn("change_weight_by_ckpt_snapshot", text)
        self.assertIn("normalize_per_timestamp", text)
        self.assertIn("eps_delta_mse_by_transition", text)
        self.assertIn("eps_to_ref_cosine_by_transition", text)
        self.assertIn("eps_initial_to_ref_cosine_by_transition", text)
        self.assertIn("eps_path_straightness_by_snapshot", text)
        self.assertIn("eps_to_ref_progress_by_transition", text)

    def test_full_dim_term_lds_analyzer_reports_sign_and_term_ablation(self):
        text = (ROOT / "common" / "analyze_full_dim_term_lds.py").read_text()
        self.assertIn("all_terms_sign_m1", text)
        self.assertIn("all_terms_sign_p1", text)
        self.assertIn("checkpoint_lds.csv", text)
        self.assertIn("snapshot_lds.csv", text)
        self.assertIn("checkpoint_snapshot_lds.csv", text)

    def test_term_weight_lds_comparison_reports_correlations(self):
        text = (ROOT / "common" / "compare_term_weight_with_lds.py").read_text()
        self.assertIn("checkpoint_lds_with_weight.csv", text)
        self.assertIn("snapshot_lds_with_weight.csv", text)
        self.assertIn("checkpoint_snapshot_lds_with_weight.csv", text)
        self.assertIn("term_spearman_lds_weight", text)
        self.assertIn("mean_weight_negative_term_lds", text)

    def test_term_sign_pattern_summary_reports_checkpoint_snapshot_quadrants(self):
        text = (ROOT / "common" / "summarize_term_sign_patterns.py").read_text()
        self.assertIn("quadrant_summary.csv", text)
        self.assertIn("checkpoint_cross_snapshot_sign_summary.csv", text)
        self.assertIn("snapshot_cross_checkpoint_sign_summary.csv", text)
        self.assertIn("checkpoint_snapshot_lds_heatmap.svg", text)
        self.assertIn("positive_snapshot_runs", text)

    def test_lds_pred_true_mismatch_diagnostic_reports_rank_errors(self):
        text = (ROOT / "common" / "diagnose_lds_pred_true_mismatch.py").read_text()
        self.assertIn("subset_pred_true_mismatch.csv", text)
        self.assertIn("true_f_quartile_summary.csv", text)
        self.assertIn("pred_true_scatter.svg", text)
        self.assertIn("pred_std_over_true_f_std", text)
        self.assertIn("worst_rank_mismatches", text)

    def test_query_update_alignment_reports_parameter_space_direction(self):
        text = (ROOT / "common" / "query_update_alignment.py").read_text()
        self.assertIn("theta_c - theta_{c+1}", text)
        self.assertIn("query_update_alignment_by_transition", text)
        self.assertIn("make_query_grad_chunk_fn", text)
        self.assertIn("descent_update = tree_sub(current_params, next_params)", text)

        analysis_text = (ROOT / "common" / "analyze_query_update_alignment.py").read_text()
        self.assertIn("transition_alignment_summary.csv", analysis_text)
        self.assertIn("snapshot_alignment_summary.csv", analysis_text)
        self.assertIn("query_update_alignment_summary.json", analysis_text)

    def test_predicted_noise_ref_process_analyzer_reports_convergence(self):
        text = (ROOT / "common" / "analyze_predicted_noise_ref_process.py").read_text()
        self.assertIn("eps_ref_mse_by_ckpt_snapshot", text)
        self.assertIn("checkpoint_ref_mse.csv", text)
        self.assertIn("snapshot_ref_mse_trends.csv", text)
        self.assertIn("eps_ref_mse_heatmap.svg", text)
        self.assertIn("transition_direction_summary.csv", text)
        self.assertIn("snapshot_direction_summary.csv", text)
        self.assertIn("eps_to_ref_cosine_heatmap.svg", text)
        self.assertIn("eps_initial_to_ref_cosine_heatmap.svg", text)
        self.assertIn("path_straightness_mean", text)
        self.assertIn("end_over_start", text)

    def test_lds_eval_supports_trajectory_state_mse_target(self):
        lds_text = (LEGACY / "LDS" / "DM_cifar_lds.py").read_text()
        self.assertIn('"trajectory_state_mse"', lds_text)
        self.assertIn("def _sample_model_space_trajectory", lds_text)
        self.assertIn("def _trajectory_state_mse", lds_text)
        self.assertIn("per_snapshot_mean", lds_text)
        self.assertIn("Generated trajectory shape", lds_text)

        common_eval = (ROOT / "common" / "lds_model_eval.py").read_text()
        self.assertIn('"trajectory_state_mse"', common_eval)

        stream_eval = (ROOT / "common" / "fast_lds_stream_score_eval.py").read_text()
        self.assertIn('"trajectory_state_mse"', stream_eval)

        full_eval = (ROOT / "common" / "full_dim_term_weighted_lds_eval.py").read_text()
        self.assertIn('"trajectory_state_mse"', full_eval)

    def test_cached_lds_seed_runner_evaluates_projected_and_full_traj_tracin(self):
        text = (ROOT / "common" / "eval_traj_tracin_cached_lds_seeds.py").read_text()
        self.assertIn("fast_lds_stream_score_eval.py", text)
        self.assertIn("full_dim_term_weighted_lds_eval.py", text)
        self.assertIn("projected_stream_cached_lds", text)
        self.assertIn("full_dim_cached_lds", text)
        self.assertIn("lds_seed_", text)
        self.assertIn("stream_scores_merged.npz", text)
        self.assertIn("full_dim_term_scores.npz", text)
        self.assertIn("--skip-missing", text)
        self.assertIn("--queries", text)

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

    def test_fast_lds_score_eval_reuses_cached_targets(self):
        text = (ROOT / "common" / "fast_lds_score_eval.py").read_text()
        self.assertIn("--target-results", text)
        self.assertIn("Existing lds_results.csv with true_f and subset_dir", text)
        self.assertIn('source_row["subset_dir"]', text)
        self.assertIn('row["pred_sum_tau"] = sum_scores(', text)
        self.assertIn('float(row["true_f"])', text)
        self.assertIn('"target_cache": str(target_path)', text)
        self.assertIn("combine_attribution_scores(", text)
        self.assertIn("plot_scatter(", text)
        self.assertNotIn("CifarTargetEvaluator", text)
        self.assertNotIn("evaluator.evaluate", text)


if __name__ == "__main__":
    unittest.main()
