from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "celeba"
    / "lds"
    / "run_training_multi_gpu.py"
)
SPEC = importlib.util.spec_from_file_location("celeba_lds_multi_gpu", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_four_gpu_partition_covers_64_subsets_once():
    chunks = MODULE.subset_chunks(64, 4)
    assert chunks == [
        list(range(0, 64, 4)),
        list(range(1, 64, 4)),
        list(range(2, 64, 4)),
        list(range(3, 64, 4)),
    ]
    assert sorted(value for chunk in chunks for value in chunk) == list(range(64))


def test_gpu_parser_rejects_duplicates():
    assert MODULE.parse_gpus("0, 1 2,3") == ["0", "1", "2", "3"]
    try:
        MODULE.parse_gpus("0,0")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate GPU ids must be rejected")
