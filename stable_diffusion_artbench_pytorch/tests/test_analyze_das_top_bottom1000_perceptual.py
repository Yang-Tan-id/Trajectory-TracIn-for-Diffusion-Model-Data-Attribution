from __future__ import annotations

from pathlib import Path

import numpy as np

from sd_artbench.analyze_das_top_bottom1000_endpoint_perceptual import select_extremes


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "src/sd_artbench/analyze_das_top_bottom1000_endpoint_perceptual.py"
)


def test_select_extremes_uses_largest_and_smallest_scores():
    scores = np.arange(5000, dtype=np.float64)
    indices = np.arange(10_000, 15_000, dtype=np.int64)
    top, bottom = select_extremes(scores, indices, 1000)
    assert set(top.tolist()) == set(range(14_000, 15_000))
    assert set(bottom.tolist()) == set(range(10_000, 11_000))
    assert not (set(top.tolist()) & set(bottom.tolist()))


def test_script_contract_is_lambda1000_and_writes_paired_report():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "scores/lambda_1000/scores.npz" in text
    assert 'default=list(range(60))' in text
    assert '(query, "top")' in text
    assert '(query, "bottom")' in text
    assert 'output_dir / "per_query.csv"' in text
    assert 'output_dir / "summary.csv"' in text
    assert 'output_dir / "report.txt"' in text
