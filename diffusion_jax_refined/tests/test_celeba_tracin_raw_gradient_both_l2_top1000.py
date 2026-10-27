from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "diffusion_jax_refined/celeba/script/run_endpoint_top500_removal.py"
RUNNER = REPO / (
    "diffusion_jax_refined/celeba/script/"
    "run_endpoint_tracin_raw_gradient_both_l2_full10k_top1000_q0_9_4gpu.sh"
)
METHOD = "endpoint_tracin_raw_gradient_linear_both_l2_full10k_pos"


class CelebATracinRawGradientBothL2Top1000Test(unittest.TestCase):
    def test_worker_uses_raw_gradient_full10k_both_l2_descending(self) -> None:
        source = WORKER.read_text()
        self.assertIn(f'"{METHOD}"', source)
        self.assertIn(
            '"celeba_endpoint_tracin_raw_gradient_"',
            source,
        )
        self.assertIn(
            'f"traj_tracin_{tracin_raw_gradient_namespace}_full10000"',
            source,
        )
        self.assertIn('"score_query_train_l2_normalized",\n            +1,', source)
        self.assertIn(
            '"query+train-L2 score descending"',
            source,
        )

    def test_runner_contract(self) -> None:
        source = RUNNER.read_text()
        self.assertIn("GPU_IDS:-4,5,6,7", source)
        self.assertIn("QUERY_IDS:-0,1,2,3,4,5,6,7,8,9", source)
        self.assertIn(f'method="{METHOD}"', source)
        self.assertIn("topk=1000", source)
        self.assertIn('output_group="endpoint_top1000_removal_q0_9"', source)
        self.assertIn("--stage validate", source)
        self.assertIn("--stage all", source)
        self.assertIn("query_index+=4", source)

    def test_runner_shell_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
