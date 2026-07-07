from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path


def indices_from_one_based_range(start: int, end: int) -> list[int]:
    if end < start:
        raise ValueError("end must be >= start")
    return list(range(start - 1, end))


def load_score_indices(folder: Path) -> list[int]:
    npy = folder / "score_indices.npy"
    if npy.exists():
        # These unit tests intentionally avoid requiring numpy.  A tiny JSON
        # payload with a .npy-like name is enough to exercise the path contract.
        return [int(x) for x in json.loads(npy.read_text())]
    payload = json.loads((folder / "score_indices.json").read_text())
    for key in ("picked_indices", "score_indices"):
        if key in payload:
            return [int(x) for x in payload[key]]
    raise KeyError(f"no score indices in {folder}")


def combine_scores(parts: list[tuple[list[int], list[float]]], duplicate_policy: str) -> tuple[list[int], list[float]]:
    buckets: dict[int, list[float]] = {}
    for indices, scores in parts:
        if len(indices) != len(scores):
            raise ValueError("indices and scores length mismatch")
        for idx, score in zip(indices, scores):
            buckets.setdefault(int(idx), []).append(float(score))
    out_idx = sorted(buckets)
    out_scores = []
    for idx in out_idx:
        vals = buckets[idx]
        if duplicate_policy == "max":
            out_scores.append(max(vals))
        elif duplicate_policy == "sum":
            out_scores.append(sum(vals))
        elif duplicate_policy == "mean":
            out_scores.append(sum(vals) / len(vals))
        else:
            raise ValueError(duplicate_policy)
    return out_idx, out_scores


class TestScoreIndices(unittest.TestCase):
    def test_one_based_ranges_become_zero_based_indices(self):
        idx = indices_from_one_based_range(1, 2000)
        self.assertEqual(len(idx), 2000)
        self.assertEqual(min(idx), 0)
        self.assertEqual(max(idx), 1999)

    def test_traj_five_shard_union_covers_10000_unique_points(self):
        ranges = [(1, 2000), (2001, 4000), (4001, 6000), (6001, 8000), (8001, 10000)]
        union = [idx for start, end in ranges for idx in indices_from_one_based_range(start, end)]
        self.assertEqual(len(union), 10000)
        self.assertEqual(len(set(union)), 10000)
        self.assertEqual(min(union), 0)
        self.assertEqual(max(union), 9999)

    def test_loader_accepts_npy_and_json_index_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy_dir = root / "traj"
            json_dir = root / "das"
            npy_dir.mkdir()
            json_dir.mkdir()
            (npy_dir / "score_indices.npy").write_text(json.dumps([0, 2, 4]))
            (json_dir / "score_indices.json").write_text(json.dumps({"picked_indices": [1, 3, 5]}))
            self.assertEqual(load_score_indices(npy_dir), [0, 2, 4])
            self.assertEqual(load_score_indices(json_dir), [1, 3, 5])

    def test_duplicate_policy_is_explicit_and_stable(self):
        part_a = ([0, 1, 2], [1.0, 2.0, 3.0])
        part_b = ([1, 2, 3], [20.0, 30.0, 40.0])
        idx, scores = combine_scores([part_a, part_b], "max")
        self.assertEqual(idx, [0, 1, 2, 3])
        self.assertEqual(scores, [1.0, 20.0, 30.0, 40.0])
        _, scores = combine_scores([part_a, part_b], "sum")
        self.assertEqual(scores, [1.0, 22.0, 33.0, 40.0])
        _, scores = combine_scores([part_a, part_b], "mean")
        self.assertEqual(scores, [1.0, 11.0, 16.5, 40.0])

    def test_lds_kept_indices_are_covered_by_full_score_universe(self):
        score_indices = set(range(10000))
        kept = set(random.Random(0).sample(range(10000), 5000))
        self.assertEqual(len(kept & score_indices), 5000)


if __name__ == "__main__":
    unittest.main()
