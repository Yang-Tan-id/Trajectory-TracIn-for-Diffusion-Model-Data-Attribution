from __future__ import annotations

import os
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


def eval_root(dataset_name: str, experiment_tag: str, metric_name: str) -> Path:
    return experiment_root(dataset_name, experiment_tag) / "eval" / metric_name


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
