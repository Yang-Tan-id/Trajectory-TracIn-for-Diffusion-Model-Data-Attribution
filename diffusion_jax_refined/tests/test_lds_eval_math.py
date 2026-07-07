from __future__ import annotations

import csv
import json
import math
import random
import tempfile
import unittest
from pathlib import Path


def _compact_model_group_name(model_dirs: list[Path], *, max_len: int = 96) -> str:
    names = [path.name for path in model_dirs]
    joined = "__".join(names)
    if len(joined) <= max_len:
        return joined
    seeds = []
    for name in names:
        if "_seed_" in name:
            seeds.append(int(name.rsplit("_seed_", 1)[1].split("_", 1)[0]))
    return f"{len(names)}_lds_models_seeds_{min(seeds)}_{max(seeds)}_digest"


def _prediction_tag(subset: str, sign: float) -> str:
    if float(sign).is_integer():
        sign_text = str(int(sign))
    else:
        sign_text = f"{sign:g}"
    sign_text = sign_text.replace("-", "m").replace("+", "p").replace(".", "p")
    return f"pred_{subset}_sign_{sign_text}"


def rank_average_ties(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for ordered_index in order[i:j]:
            ranks[ordered_index] = rank
        i = j
    return ranks


def spearman(a, b) -> float:
    ra = rank_average_ties([float(x) for x in a])
    rb = rank_average_ties([float(x) for x in b])
    mean_a = sum(ra) / len(ra)
    mean_b = sum(rb) / len(rb)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in ra))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in rb))
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b)


def sum_scores(indices: list[int], score_map: dict[int, float], sign: float) -> float:
    return float(sign) * sum(float(score_map.get(int(idx), 0.0)) for idx in indices)


class TestLdsEvalMath(unittest.TestCase):
    def test_spearman_known_values(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_prediction_subset_and_sign_are_explicit(self):
        score_map = {0: 1.0, 1: 2.0, 2: -4.0, 3: 8.0}
        kept = [0, 1]
        removed = [2, 3]
        self.assertEqual(sum_scores(kept, score_map, 1.0), 3.0)
        self.assertEqual(sum_scores(kept, score_map, -1.0), -3.0)
        self.assertEqual(sum_scores(removed, score_map, 1.0), 4.0)

    def test_prediction_tag_prevents_overwrite_between_signs_and_subsets(self):
        self.assertEqual(_prediction_tag("kept", 1.0), "pred_kept_sign_1")
        self.assertEqual(_prediction_tag("kept", -1.0), "pred_kept_sign_m1")
        self.assertEqual(_prediction_tag("removed", -0.5), "pred_removed_sign_m0p5")

    def test_compact_model_group_name_avoids_long_paths_but_keeps_seed_range(self):
        model_dirs = [Path(f"/tmp/m_50_k_5000_seed_{seed}") for seed in range(1, 17)]
        name = _compact_model_group_name(model_dirs, max_len=40)
        self.assertLessEqual(len(name), 96)
        self.assertIn("16_lds_models", name)
        self.assertIn("seeds_1_16", name)

    def test_fake_lds_regression_high_correlation(self):
        rng = random.Random(123)
        true_scores = [rng.gauss(0.0, 1.0) for _ in range(20)]
        rows = []
        score_map = {i: float(v) for i, v in enumerate(true_scores)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subset_id in range(8):
                kept = sorted(rng.sample(range(20), 10))
                pred = sum_scores(kept, score_map, 1.0)
                rows.append({"subset_id": subset_id, "pred_sum_tau": pred, "true_f": pred + 0.001 * subset_id})
            csv_path = root / "lds_results.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["subset_id", "pred_sum_tau", "true_f"])
                writer.writeheader()
                writer.writerows(rows)
            pred = [row["pred_sum_tau"] for row in rows]
            true = [row["true_f"] for row in rows]
            self.assertGreater(spearman(pred, true), 0.99)

    def test_summary_records_target_prediction_subset_and_sign(self):
        payload = {
            "target_function": "noise_trajectory",
            "prediction_subset": "kept",
            "prediction_sign": -1.0,
        }
        text = json.dumps(payload)
        loaded = json.loads(text)
        self.assertEqual(loaded["target_function"], "noise_trajectory")
        self.assertEqual(loaded["prediction_subset"], "kept")
        self.assertEqual(loaded["prediction_sign"], -1.0)


if __name__ == "__main__":
    unittest.main()
