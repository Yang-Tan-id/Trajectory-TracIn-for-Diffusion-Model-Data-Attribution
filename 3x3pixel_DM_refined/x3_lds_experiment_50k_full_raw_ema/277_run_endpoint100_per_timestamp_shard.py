"""Run all 100 trajectory timestamps with per-t inverse-noise alignment."""

import importlib.util
import sys
from pathlib import Path

from exp_config import ATTR_DIR, DAS_TIMESTEPS


def load_worker():
    path = Path(__file__).with_name(
        "271_run_endpoint20_meanloss_pairing_shard.py"
    )
    spec = importlib.util.spec_from_file_location("endpoint100_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    worker = load_worker()
    worker.E20_TIMESTAMP_POSITIONS = tuple(range(len(DAS_TIMESTEPS)))
    worker.E20_TIMESTEPS = tuple(DAS_TIMESTEPS)
    worker.E20_MODES = ("per_timestamp_aligned",)
    worker.E20_VERSION = 100

    def shard_root(checkpoint_shard_index, checkpoint_shard_count):
        return (
            ATTR_DIR
            / "_tracin_das_endpoint100_inverse_noise_per_timestamp_10q_shards"
            / f"checkpoint_shard_{int(checkpoint_shard_index):02d}_of_"
              f"{int(checkpoint_shard_count):02d}"
        )

    worker.e20_shard_root = shard_root
    worker.main()


if __name__ == "__main__":
    main()
