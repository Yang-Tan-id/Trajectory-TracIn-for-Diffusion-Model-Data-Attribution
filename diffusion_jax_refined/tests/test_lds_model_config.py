from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path


def write_fake_lds_model(
    root: Path,
    *,
    m: int,
    k: int,
    universe_size: int,
    lds_seed: int,
    train_seed: int = 42,
    sample_model_mode: str = "prompted_solo",
    dataset_percentage: float | None = None,
) -> Path:
    run_parts = [f"m_{m}", f"k_{k}"]
    if dataset_percentage is not None:
        run_parts.append(f"pct_{dataset_percentage:g}".replace(".", "p"))
    run_parts.append(f"subset_seed_{lds_seed}")
    model_dir = root / sample_model_mode / f"train_seed_{train_seed}" / "_".join(run_parts)
    models_dir = model_dir / "models"
    rng = random.Random(lds_seed)
    subsets = []
    for subset_id in range(m):
        subset_dir = models_dir / f"subset_{subset_id:04d}"
        subset_dir.mkdir(parents=True)
        kept = sorted(rng.sample(range(universe_size), k))
        excluded = sorted(set(range(universe_size)) - set(kept))
        (subset_dir / "kept_attribution_indices.npy").write_text(json.dumps(kept))
        (subset_dir / "excluded_attribution_indices.npy").write_text(json.dumps(excluded))
        (subset_dir / "subset_metadata.json").write_text(
            json.dumps({"subset_id": subset_id, "subset_size": len(kept), "dataset_percentage": dataset_percentage})
        )
        (subset_dir / "train_config.json").write_text(json.dumps({"train_config": {"seed": train_seed}}))
        (subset_dir / f"seed_{train_seed}_epoch_0200.ckpt").write_text("fake")
        subsets.append({"subset_id": subset_id, "subset_seed": rng.randrange(0, 2**31 - 1)})
    (model_dir / "lds_model_config.json").write_text(
        json.dumps(
            {
                "dataset": "cifar2",
                "mode": "prompted",
                "sample_model_mode": sample_model_mode,
                "model_train_seed": train_seed,
                "m": m,
                "k": k,
                "dataset_percentage": dataset_percentage,
                "dataset_universe_size": universe_size,
                "sample_random_seed": lds_seed,
                "complete": True,
                "subsets": subsets,
                "base_checkpoint": "/fake/seed_42_epoch_0200.ckpt",
                "train_config_template": {"seed": train_seed, "timesteps": 1000},
            }
        )
    )
    return model_dir


class TestLdsModelConfig(unittest.TestCase):
    def test_k5000_subset_files_are_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_fake_lds_model(Path(tmp), m=2, k=5000, universe_size=10000, lds_seed=1)
            cfg = json.loads((model_dir / "lds_model_config.json").read_text())
            self.assertTrue(cfg["complete"])
            self.assertEqual(len(cfg["subsets"]), 2)
            for subset_id in range(2):
                subset_dir = model_dir / "models" / f"subset_{subset_id:04d}"
                kept = json.loads((subset_dir / "kept_attribution_indices.npy").read_text())
                excluded = json.loads((subset_dir / "excluded_attribution_indices.npy").read_text())
                self.assertEqual(len(kept), 5000)
                self.assertEqual(len(excluded), 5000)
                self.assertEqual(len(set(kept) & set(excluded)), 0)
                self.assertEqual(len(set(kept) | set(excluded)), 10000)

    def test_k8000_subset_size_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_fake_lds_model(Path(tmp), m=1, k=8000, universe_size=10000, lds_seed=1)
            subset_dir = model_dir / "models" / "subset_0000"
            self.assertEqual(len(json.loads((subset_dir / "kept_attribution_indices.npy").read_text())), 8000)
            self.assertEqual(len(json.loads((subset_dir / "excluded_attribution_indices.npy").read_text())), 2000)

    def test_model_train_seed_and_subset_seed_have_separate_folder_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_fake_lds_model(
                Path(tmp),
                m=1,
                k=10,
                universe_size=20,
                lds_seed=7,
                train_seed=123,
                sample_model_mode="prompted_multi",
            )
            self.assertEqual(model_dir.parts[-3:], ("prompted_multi", "train_seed_123", "m_1_k_10_subset_seed_7"))
            subset_dir = model_dir / "models" / "subset_0000"
            train_cfg = json.loads((subset_dir / "train_config.json").read_text())
            self.assertEqual(train_cfg["train_config"]["seed"], 123)
            self.assertTrue((subset_dir / "seed_123_epoch_0200.ckpt").is_file())
            cfg = json.loads((model_dir / "lds_model_config.json").read_text())
            self.assertEqual(cfg["sample_model_mode"], "prompted_multi")
            self.assertEqual(cfg["model_train_seed"], 123)
            self.assertEqual(cfg["sample_random_seed"], 7)

    def test_dataset_percentage_is_recorded_with_subset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_fake_lds_model(
                Path(tmp),
                m=1,
                k=25,
                universe_size=100,
                lds_seed=9,
                train_seed=42,
                dataset_percentage=25,
            )
            self.assertTrue(model_dir.name.startswith("m_1_k_25_pct_25_subset_seed_9"))
            cfg = json.loads((model_dir / "lds_model_config.json").read_text())
            self.assertEqual(cfg["dataset_percentage"], 25)
            self.assertEqual(cfg["k"], 25)
            subset_meta = json.loads((model_dir / "models" / "subset_0000" / "subset_metadata.json").read_text())
            self.assertEqual(subset_meta["dataset_percentage"], 25)


if __name__ == "__main__":
    unittest.main()
