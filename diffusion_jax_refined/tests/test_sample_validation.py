from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def validate_fake_sample(sample_dir: Path, *, expected_prompt: str, expected_seed: int) -> None:
    required = ["trajectory_xt.npy", "trajectory_t.npy", "final_state.npy", "seed_info.json"]
    missing = [name for name in required if not (sample_dir / name).is_file()]
    manifest = sample_dir.parent / "manifest.json"
    if not manifest.is_file():
        missing.append("manifest.json")
    if missing:
        raise FileNotFoundError(f"missing {missing}")
    seed_info = json.loads((sample_dir / "seed_info.json").read_text())
    if int(seed_info.get("seed", -1)) != expected_seed:
        raise ValueError("seed mismatch")
    if str(seed_info.get("prompt")) != expected_prompt:
        raise ValueError("prompt mismatch")
    manifest_payload = json.loads(manifest.read_text())
    if expected_seed not in [int(x) for x in manifest_payload.get("seeds", [])]:
        raise ValueError("manifest seed mismatch")
    trajectory_shape = json.loads((sample_dir / "trajectory_xt.npy").read_text())["shape"]
    times_shape = json.loads((sample_dir / "trajectory_t.npy").read_text())["shape"]
    final_state_shape = json.loads((sample_dir / "final_state.npy").read_text())["shape"]
    if len(trajectory_shape) != 5:
        raise ValueError("trajectory must be (K,B,H,W,C)")
    if len(final_state_shape) != 4:
        raise ValueError("final state must be (B,H,W,C)")
    if trajectory_shape[0] != times_shape[0]:
        raise ValueError("trajectory/time length mismatch")
    if trajectory_shape[1] != final_state_shape[0]:
        raise ValueError("trajectory/final batch mismatch")


class TestSampleValidation(unittest.TestCase):
    def make_sample(self, root: Path, *, prompt: str = "horse", seed: int = 42) -> Path:
        run = root / "model_prompted"
        sample = run / f"seed_{seed:06d}"
        sample.mkdir(parents=True)
        (sample / "trajectory_xt.npy").write_text(json.dumps({"shape": [3, 1, 4, 4, 3]}))
        (sample / "trajectory_t.npy").write_text(json.dumps({"shape": [3]}))
        (sample / "final_state.npy").write_text(json.dumps({"shape": [1, 4, 4, 3]}))
        (sample / "seed_info.json").write_text(json.dumps({"seed": seed, "prompt": prompt}))
        (run / "manifest.json").write_text(json.dumps({"seeds": [seed], "prompt": prompt}))
        return sample

    def test_valid_sample_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = self.make_sample(Path(tmp), prompt="horse", seed=42)
            validate_fake_sample(sample, expected_prompt="horse", expected_seed=42)

    def test_prompt_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = self.make_sample(Path(tmp), prompt="horse", seed=42)
            with self.assertRaises(ValueError):
                validate_fake_sample(sample, expected_prompt="automobile", expected_seed=42)

    def test_seed_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = self.make_sample(Path(tmp), prompt="horse", seed=42)
            with self.assertRaises(ValueError):
                validate_fake_sample(sample, expected_prompt="horse", expected_seed=0)

    def test_shape_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = self.make_sample(Path(tmp), prompt="horse", seed=42)
            (sample / "trajectory_t.npy").write_text(json.dumps({"shape": [2]}))
            with self.assertRaises(ValueError):
                validate_fake_sample(sample, expected_prompt="horse", expected_seed=42)


if __name__ == "__main__":
    unittest.main()
