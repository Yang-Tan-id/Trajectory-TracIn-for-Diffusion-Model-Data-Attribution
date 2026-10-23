from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGER = (
    REPO_ROOT
    / "diffusion_jax_refined/3dshapes/script/package_endpoint_q0_99_scores_lds.py"
)
LAUNCHER = (
    REPO_ROOT
    / "diffusion_jax_refined/3dshapes/tacc/spr/"
    "run_package_endpoint_q0_99_scores_lds_spr.sh"
)
SUBMITTER = (
    REPO_ROOT
    / "diffusion_jax_refined/3dshapes/tacc/skx_dev/"
    "submit_package_endpoint_q0_99_scores_lds_skx_dev.sh"
)


class Shape3DResultPackagerTest(unittest.TestCase):
    def test_synthetic_score_and_lds_export_completes_with_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_root = root / "result" / "experiment1"
            output_dir = root / "export"
            query_file = root / "queries.json"
            query_file.write_text(
                json.dumps(
                    {
                        "queries": [
                            {
                                "id": 0,
                                "prompt": "shape_cube",
                                "initial_seed": 100,
                            }
                        ]
                    }
                )
            )

            query_rel = Path("query_shape_cube") / "initial_seed_100"
            score_namespace = (
                "traj_tracin_recreate_raw_polluted_endpoint_delta_raw_"
                "timestamp_sum_squared_q0_99"
            )
            score_dir = (
                result_root
                / "attribution_score/prompted_solo/train_seed_42"
                / query_rel
                / score_namespace
                / "score_raw"
            )
            score_dir.mkdir(parents=True)
            (score_dir / "scores.npy").write_bytes(b"synthetic-score")
            (score_dir / "score_indices.npy").write_bytes(b"synthetic-indices")
            (score_dir / "projected_gradients.npz").write_bytes(
                b"must-not-be-exported"
            )

            lds_namespace = f"{score_namespace}_raw"
            summary_dir = (
                result_root
                / "eval/prompted_solo"
                / query_rel
                / "lds"
                / lds_namespace
                / "simple_loss/pred_kept_sign_-1/subset_00"
            )
            summary_dir.mkdir(parents=True)
            target_cache = (
                result_root
                / "eval/prompted_solo/lds_target_cache/simple_loss/true_f_results.csv"
            )
            target_cache.parent.mkdir(parents=True)
            target_cache.write_text("true_f\n0.5\n")
            (summary_dir / "lds_summary.json").write_text(
                json.dumps(
                    {
                        "lds_spearman": 0.25,
                        "lds_percent": 25.0,
                        "target_cache": str(target_cache),
                    }
                )
            )
            (summary_dir / "lds_results.csv").write_text(
                "true_f,pred_sum_tau\n0.5,0.25\n"
            )
            (summary_dir / "predictions.npy").write_bytes(
                b"intermediate-predictions-must-not-be-exported"
            )
            invalid_summary = (
                result_root
                / "eval/prompted_solo"
                / query_rel
                / "lds"
                / lds_namespace
                / "noise_trajectory/pred_kept_sign_-1/subset_00/lds_summary.json"
            )
            invalid_summary.parent.mkdir(parents=True)
            invalid_summary.write_text("")

            unrelated = result_root / "model" / "must_not_be_selected.ckpt"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_bytes(b"checkpoint")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGER),
                    "--query-file",
                    str(query_file),
                    "--query-ids",
                    "0",
                    "--result-root",
                    str(result_root),
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("[startup] queries=1", completed.stdout)
            self.assertIn("[scan] query=0 (1/1)", completed.stdout)
            self.assertIn("[warning] skipping invalid LDS summary", completed.stdout)
            self.assertIn("[metadata]", completed.stdout)
            self.assertIn("[archive progress]", completed.stdout)
            self.assertIn("[saved]", completed.stdout)
            self.assertTrue(
                (output_dir / "shape3d_q0_99_endpoint_scores_lds.tar.gz").is_file()
                or (output_dir / "shape3d_q0_99_endpoint_scores_lds.tar.zst").is_file()
            )

            inventory_path = output_dir / "metadata/score_inventory.csv"
            with inventory_path.open(newline="") as handle:
                inventory = list(csv.DictReader(handle))
            self.assertEqual(len(inventory), 1)
            self.assertEqual(inventory[0]["family"], "endpoint_pollute")
            self.assertEqual(inventory[0]["transform"], "raw_gradient")
            self.assertEqual(inventory[0]["file_count"], "2")

            lds_path = output_dir / "metadata/lds_all_results.csv"
            with lds_path.open(newline="") as handle:
                lds_rows = list(csv.DictReader(handle))
            self.assertEqual(len(lds_rows), 1)
            self.assertEqual(lds_rows[0]["variant"], "raw")
            self.assertEqual(lds_rows[0]["lds_percent"], "25.0")

            invalid_report = (
                output_dir / "metadata/invalid_lds_summaries.txt"
            ).read_text()
            self.assertIn("noise_trajectory", invalid_report)
            self.assertIn("JSONDecodeError", invalid_report)

            selected = (
                output_dir / "metadata/selected_result_directories.txt"
            ).read_text()
            self.assertIn(score_namespace, selected)
            self.assertIn(lds_namespace, selected)
            self.assertNotIn("must_not_be_selected.ckpt", selected)

            selected_files = (
                output_dir / "metadata/selected_result_files.txt"
            ).read_text()
            self.assertIn("scores.npy", selected_files)
            self.assertIn("score_indices.npy", selected_files)
            self.assertIn("lds_summary.json", selected_files)
            self.assertIn("lds_results.csv", selected_files)
            self.assertIn("true_f_results.csv", selected_files)
            self.assertNotIn("projected_gradients.npz", selected_files)
            self.assertNotIn("predictions.npy", selected_files)

    def test_launcher_has_fail_fast_inputs_and_unbuffered_progress(self) -> None:
        text = LAUNCHER.read_text()
        self.assertIn('query_file="${QUERY_FILE:-', text)
        self.assertIn('[[ -r "$query_file" ]]', text)
        self.assertIn('mkdir -p "$output_dir"', text)
        self.assertIn('PYTHONUNBUFFERED=1 "$python_bin" -u', text)
        self.assertIn('--query-file "$query_file"', text)

    def test_stampede3_submitter_never_uses_explicit_export(self) -> None:
        text = SUBMITTER.read_text()
        self.assertIn("-p skx-dev", text)
        self.assertIn('export REPO_ROOT="$repo"', text)
        self.assertIn('export OUTPUT_DIR="$output_dir"', text)
        self.assertIn('--chdir="$SCRATCH"', text)
        self.assertNotIn("--export", text)


if __name__ == "__main__":
    unittest.main()
