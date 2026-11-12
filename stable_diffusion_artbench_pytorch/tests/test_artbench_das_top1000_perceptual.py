from pathlib import Path

import numpy as np

from sd_artbench.analyze_das_vs_pint_top1000_endpoint_perceptual import (
    das_score_files,
    summary,
)


def test_das_score_files_sorts_numeric_lambdas(tmp_path: Path) -> None:
    for value in (10, 0.5, 2):
        path = tmp_path / f"lambda_{value:g}" / "scores.npz"
        path.parent.mkdir()
        np.savez(path, scores=np.zeros((1, 1)))
    assert [value for value, _ in das_score_files(tmp_path, None)] == [0.5, 2, 10]


def test_das_score_files_requires_requested_lambda(tmp_path: Path) -> None:
    try:
        das_score_files(tmp_path, [200])
    except FileNotFoundError as error:
        assert "200" in str(error)
    else:
        raise AssertionError("missing lambda should fail")


def test_summary() -> None:
    assert summary(np.asarray([1.0, 2.0, 3.0])) == (2.0, 2.0, np.std([1.0, 2.0, 3.0]))
