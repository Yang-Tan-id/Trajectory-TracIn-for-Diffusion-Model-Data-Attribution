#!/usr/bin/env python3
"""Run the generic CelebA removal worker with the fixed EMA-DAS contract."""

from __future__ import annotations

import run_endpoint_top500_removal as worker


METHOD = "das_ema_lambda200_full10k_pos"
_base_score_spec = worker.score_spec


def score_spec(*, result_root, train_seed, record, method):
    if method != METHOD:
        return _base_score_spec(
            result_root=result_root,
            train_seed=train_seed,
            record=record,
            method=method,
        )

    root = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{worker._prompt_path_tag(str(record['prompt']))}"
        / f"initial_seed_{worker.query_seed(record)}"
    )
    return (
        root / "das_50x1_mc1_full10k_ema" / "score" / "lambda_200",
        +1,
        (
            "stored full-10000 EMA-DAS 50x1 lambda=200 score descending; "
            "lambda selected once by mean Q0-Q99 simple_loss_50x1 LDS"
        ),
        +1.0,
    )


if __name__ == "__main__":
    worker.METHODS = (*worker.METHODS, METHOD)
    worker.score_spec = score_spec
    worker.main()
