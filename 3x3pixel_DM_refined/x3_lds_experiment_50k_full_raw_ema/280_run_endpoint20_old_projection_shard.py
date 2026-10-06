"""Run endpoint20 per-t alignment with the old trajectory projection seed."""

import importlib.util
from pathlib import Path

from exp_config import ATTR_DIR


def load_worker():
    path = Path(__file__).with_name(
        "271_run_endpoint20_meanloss_pairing_shard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "endpoint20_old_projection_worker", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    worker = load_worker()
    worker.E20_MODES = ("per_timestamp_aligned",)
    worker.E20_VERSION = 201
    worker.E20_PROJECTION_NAMESPACE = "trajectory_bridge_projection"

    def shard_root(checkpoint_shard_index, checkpoint_shard_count):
        return (
            ATTR_DIR
            / "_tracin_das_endpoint20_old_projection_per_t_10q_shards"
            / f"checkpoint_shard_{int(checkpoint_shard_index):02d}_of_"
              f"{int(checkpoint_shard_count):02d}"
        )

    worker.e20_shard_root = shard_root
    worker.main()


if __name__ == "__main__":
    main()
