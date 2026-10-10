from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "celeba" / "script" / "prepare_celeba.py"
SPEC = importlib.util.spec_from_file_location("prepare_celeba", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_seeded_training_selection_is_deterministic_and_split_only():
    partitions = {f"{index:06d}.jpg": index % 3 for index in range(1, 101)}
    first = MODULE.select_training_filenames(partitions, seed=42, train_size=20)
    second = MODULE.select_training_filenames(partitions, seed=42, train_size=20)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 20
    assert all(partitions[str(filename)] == 0 for filename in first)


def test_center_crop_uses_exact_140_square_before_resize():
    rows = np.arange(218, dtype=np.uint8)[:, None]
    cols = np.arange(178, dtype=np.uint8)[None, :]
    pixels = np.zeros((218, 178, 3), dtype=np.uint8)
    pixels[..., 0] = cols
    pixels[..., 1] = rows
    image = Image.fromarray(pixels)
    cropped = MODULE.center_crop_resize(image, crop_size=140, image_size=140)
    array = np.asarray(cropped)
    assert array.shape == (140, 140, 3)
    assert int(array[0, 0, 0]) == 19
    assert int(array[0, 0, 1]) == 39
    assert int(array[-1, -1, 0]) == 158
    assert int(array[-1, -1, 1]) == 178
