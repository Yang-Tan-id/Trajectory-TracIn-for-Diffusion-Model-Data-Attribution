from __future__ import annotations

import os
import re
import sys
from pathlib import Path


REFINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REFINE_ROOT.parent
LEGACY_JAX_ROOT = REFINE_ROOT / "legacy_jax"


def add_legacy_jax_to_path() -> None:
    legacy = str(LEGACY_JAX_ROOT)
    if legacy not in sys.path:
        sys.path.insert(0, legacy)


def chdir_legacy_jax_root() -> None:
    os.chdir(LEGACY_JAX_ROOT)


def dataset_root(dataset_name: str) -> Path:
    return REFINE_ROOT / dataset_name


def experiment_root(dataset_name: str, experiment_tag: str) -> Path:
    return dataset_root(dataset_name) / "result" / experiment_tag


def model_root(dataset_name: str, experiment_tag: str) -> Path:
    return experiment_root(dataset_name, experiment_tag) / "model"


def attribution_root(dataset_name: str, experiment_tag: str, algorithm: str) -> Path:
    return experiment_root(dataset_name, experiment_tag) / "attribution_score" / algorithm


def path_tag(value: object) -> str:
    text = str(value).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "empty"


def attribution_run_root(
    dataset_name: str,
    experiment_tag: str,
    query: object,
    initial_seed: int,
    model_mode: str | None = None,
    train_seed: int | None = None,
    *,
    unprompted: bool = False,
) -> Path:
    """Folder containing every algorithm/range output for one saved query sample."""
    score_model_mode = model_mode or os.environ.get(
        "ATTRIBUTION_SCORE_MODEL_MODE",
        os.environ.get("ATTRIBUTION_SAMPLE_MODEL_MODE", os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo")),
    )
    score_train_seed = int(train_seed if train_seed is not None else os.environ.get("TRAIN_SEED", "42"))
    query_component = "unprompted" if unprompted or str(score_model_mode).startswith("unprompted_") else f"query_{path_tag(query)}"
    return (
        experiment_root(dataset_name, experiment_tag)
        / "attribution_score"
        / str(score_model_mode)
        / f"train_seed_{score_train_seed}"
        / query_component
        / f"initial_seed_{int(initial_seed)}"
    )


def eval_root(dataset_name: str, experiment_tag: str, metric_name: str) -> Path:
    return experiment_root(dataset_name, experiment_tag) / "eval" / metric_name


def eval_run_root(
    dataset_name: str,
    experiment_tag: str,
    query: object,
    initial_seed: int,
    model_mode: str | None = None,
    *,
    unprompted: bool = False,
) -> Path:
    score_model_mode = model_mode or os.environ.get(
        "ATTRIBUTION_SCORE_MODEL_MODE",
        os.environ.get("ATTRIBUTION_SAMPLE_MODEL_MODE", os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo")),
    )
    query_component = "unprompted" if unprompted or str(score_model_mode).startswith("unprompted_") else f"query_{path_tag(query)}"
    return (
        experiment_root(dataset_name, experiment_tag)
        / "eval"
        / str(score_model_mode)
        / query_component
        / f"initial_seed_{int(initial_seed)}"
    )


def ensure_experiment_dirs(dataset_name: str, experiment_tag: str) -> None:
    for path in (
        model_root(dataset_name, experiment_tag),
        experiment_root(dataset_name, experiment_tag) / "attribution_score",
        experiment_root(dataset_name, experiment_tag) / "eval",
    ):
        path.mkdir(parents=True, exist_ok=True)


def as_legacy_relative(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(p)
