import numpy as np

from sd_artbench.topk_removal import select_topk


def test_select_topk_descending_and_ascending():
    scores = np.asarray([0.5, -2.0, 3.0, 1.0], dtype=np.float64)
    indices = np.asarray([10, 11, 12, 13], dtype=np.int64)

    removed, stored, ranking = select_topk(
        scores, indices, top_k=2, ranking_sign=1
    )
    assert removed.tolist() == [12, 13]
    assert stored.tolist() == [3.0, 1.0]
    assert ranking.tolist() == [3.0, 1.0]

    removed, stored, ranking = select_topk(
        scores, indices, top_k=2, ranking_sign=-1
    )
    assert removed.tolist() == [11, 10]
    assert stored.tolist() == [-2.0, 0.5]
    assert ranking.tolist() == [2.0, -0.5]
