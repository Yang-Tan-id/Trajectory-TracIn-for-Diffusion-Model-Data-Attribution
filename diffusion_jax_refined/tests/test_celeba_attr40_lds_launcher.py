from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "celeba"
    / "lds"
    / "run_training_attr40_multi_gpu.py"
)
SPEC = importlib.util.spec_from_file_location("celeba_attr40_lds", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_four_workers_partition_all_64_models_once():
    chunks = MODULE.subset_chunks(64, 4)
    assert chunks == [list(range(offset, 64, 4)) for offset in range(4)]
    assert sorted(item for chunk in chunks for item in chunk) == list(range(64))
