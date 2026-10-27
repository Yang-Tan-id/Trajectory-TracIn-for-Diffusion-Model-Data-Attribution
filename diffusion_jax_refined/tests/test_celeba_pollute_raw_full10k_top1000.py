from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "diffusion_jax_refined/celeba/script/run_endpoint_top500_removal.py"
RUNNER = REPO / (
    "diffusion_jax_refined/celeba/script/"
    "run_endpoint_pollute_raw_full10k_top1000_q0_9_4gpu.sh"
)


class CelebAPolluteRawFull10kTop1000Test(unittest.TestCase):
    def test_worker_has_full10k_raw_contract(self) -> None:
        source = WORKER.read_text()
        self.assertIn(
            '"endpoint_pollute_adamw_timestamp_raw_full10k_pos"', source
        )
        self.assertIn(
            'root / f"traj_tracin_{pollute_namespace}_full10000" / "score",\n'
            "            +1,",
            source,
        )
        self.assertIn(
            '"raw score descending"', source
        )

    def test_runner_is_exactly_q0_q9_and_resume_safe(self) -> None:
        source = RUNNER.read_text()
        self.assertIn("QUERY_IDS:-0,1,2,3,4,5,6,7,8,9", source)
        self.assertIn("topk=1000", source)
        self.assertIn(
            'method="endpoint_pollute_adamw_timestamp_raw_full10k_pos"', source
        )
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
