from __future__ import annotations

import unittest

import numpy as np

from sd_artbench.analyze_five_method_cross_query_topk_overlap import (
    METHODS,
    score_bank,
    select_topk,
)


class FiveMethodCrossQueryTopKOverlapTest(unittest.TestCase):
    def test_five_methods_and_directions(self) -> None:
        self.assertEqual(len(METHODS), 5)
        self.assertEqual(METHODS[0][-2:], (+1, +1))
        self.assertEqual(METHODS[1][-2:], (+1, +1))
        self.assertEqual(METHODS[2][-2:], (+1, -1))
        self.assertEqual(METHODS[3][-2:], (+1, -1))
        self.assertEqual(METHODS[4][-2:], (+1, -1))

    def test_score_bank_variant_and_retrac_layouts(self) -> None:
        variant_artifact = {
            "path": "endpoint.npz",
            "scores": np.arange(2 * 4 * 3 * 5).reshape(2, 4, 3, 5),
            "indices": np.arange(5),
            "transforms": ["raw_gradient", "adamw_full"],
            "variants": ["raw", "query_l2", "train_l2", "query_train_l2"],
        }
        retrac_artifact = {
            "path": "retrac.npz",
            "scores": np.arange(2 * 3 * 5).reshape(2, 3, 5),
            "indices": np.arange(5),
            "transforms": ["raw_gradient", "adamw_full"],
            "variants": None,
        }
        self.assertEqual(
            score_bank(
                variant_artifact,
                transform="adamw_full",
                variant="query_train_l2",
            ).shape,
            (3, 5),
        )
        self.assertEqual(
            score_bank(
                retrac_artifact, transform="raw_gradient", variant=None
            ).shape,
            (3, 5),
        )

    def test_topk_respects_ranking_sign(self) -> None:
        scores = np.asarray([1.0, 3.0, 2.0, -1.0])
        indices = np.asarray([10, 11, 12, 13])
        self.assertEqual(
            select_topk(scores, indices, topk=2, ranking_sign=+1),
            {11, 12},
        )
        self.assertEqual(
            select_topk(scores, indices, topk=2, ranking_sign=-1),
            {10, 13},
        )

    def test_same_query_overlap_definition(self) -> None:
        raw = {1, 2, 3, 4}
        adamw = {3, 4, 5, 6}
        overlap_count = len(raw & adamw)
        self.assertEqual(overlap_count, 2)
        self.assertEqual(overlap_count / len(raw), 0.5)
        self.assertEqual(1.0 - overlap_count / len(raw), 0.5)
        self.assertEqual(overlap_count / len(raw | adamw), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
