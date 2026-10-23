from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO_ROOT
    / "diffusion_jax_refined/celeba/script/celeba_top1000_250_plan.py"
)
WORKER_PATH = (
    REPO_ROOT
    / "diffusion_jax_refined/celeba/script/run_endpoint_top500_removal.py"
)
RUNNER = (
    REPO_ROOT
    / "diffusion_jax_refined/celeba/tacc/h100/run_celeba_top1000_250_h100.sh"
)
SLOT = (
    REPO_ROOT
    / "diffusion_jax_refined/celeba/tacc/h100/run_celeba_top1000_250_slot.sh"
)
SUBMITTER = (
    REPO_ROOT
    / "diffusion_jax_refined/celeba/tacc/h100/submit_celeba_top1000_250_h100.sh"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = load_module("celeba_top1000_250_plan", PLAN_PATH)


class CelebATop1000250H100Test(unittest.TestCase):
    def test_task_plan_is_exactly_250_unique_models(self) -> None:
        tasks = PLAN.build_tasks()
        self.assertEqual(len(tasks), 250)
        self.assertEqual(len(set(tasks)), 250)

        for method in PLAN.BOTH_L2_METHODS:
            ids = [query_id for task_method, query_id in tasks if task_method == method]
            self.assertEqual(ids, list(range(10, 70)))
        raw_ids = [
            query_id
            for method, query_id in tasks
            if method == PLAN.RAW_METHOD
        ]
        self.assertEqual(raw_ids, list(range(70)))

    def test_transfer_manifest_contains_all_required_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            query_file = Path(temporary_directory) / "queries.json"
            query_file.write_text(
                json.dumps(
                    {
                        "queries": [
                            {
                                "id": query_id,
                                "prompt": f"attribute_{query_id}",
                                "initial_seed": 1000 + query_id,
                            }
                            for query_id in range(70)
                        ]
                    }
                )
            )
            manifest = PLAN.transfer_manifest(query_file)

        self.assertEqual(len(manifest), 896)
        text = "\n".join(map(str, manifest))
        self.assertIn("queries_attr40_val25_x4.json", text)
        self.assertIn("seed_42_epoch_0200.ckpt", text)
        self.assertIn("dataset.npz", text)
        self.assertIn("celeba_attr40_query100_lds_true_f_long.csv", text)
        self.assertIn("trajectory_xt.npy", text)
        self.assertIn("trajectory_t.npy", text)
        self.assertIn("score_query_train_l2_normalized/scores.npy", text)
        self.assertIn("linear_shape3d_semantics_full10000/score/scores.npy", text)
        self.assertEqual(sum(path.name == "manifest.json" for path in manifest), 250)

    def test_raw_full10k_score_contract_is_ascending(self) -> None:
        worker = WORKER_PATH.read_text()
        self.assertIn(
            '"endpoint_tracin_adamw_linear_raw_full10k_neg"',
            worker,
        )
        self.assertIn(
            'root / f"traj_tracin_{tracin_namespace}_full10000" / "score",\n'
            '            -1,',
            worker,
        )
        self.assertIn(
            '"stored full-10000 endpoint-TracIn linear raw score ascending",',
            worker,
        )
        self.assertIn(
            '"stored full-10000 endpoint-TracIn linear raw score ascending",\n'
            '            -1.0,',
            worker,
        )

    def test_h100_scripts_are_static_and_resume_safe(self) -> None:
        runner = RUNNER.read_text()
        slot = SLOT.read_text()
        submitter = SUBMITTER.read_text()

        self.assertIn("#SBATCH -p h100", runner)
        self.assertIn("#SBATCH -N 4", runner)
        self.assertIn("#SBATCH -n 16", runner)
        self.assertIn("#SBATCH -t 48:00:00", runner)
        self.assertIn("all 250 tasks valid", runner)
        self.assertIn("endpoint_top1000_removal_q0_69", runner)
        self.assertIn("task_index+=slot_count", slot)
        self.assertIn("--stage all", slot)
        self.assertIn("--files-from=\"$manifest\"", submitter)
        self.assertIn("[bootstrap]", submitter)
        self.assertIn("files=896", submitter)
        self.assertNotIn("--export", submitter)

        for script in (RUNNER, SLOT, SUBMITTER):
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
