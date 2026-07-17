from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from .config_loader import load_config, require_attr


STAGES = {
    "train_datapoint_gradient",
    "query_gradient",
    "score",
}


def canonical_train_model_mode(mode: str) -> str:
    if mode in ("prompted_multi", "prompted_jax", "prompted"):
        return "prompted_solo"
    if mode in ("unprompted_multi", "unprompted_jax", "unprompted"):
        return "unprompted_solo"
    return mode


def _stage_root(cfg_module: Any, algorithm: str, stage: str) -> Path:
    train_seed = int(os.environ.get("TRAIN_SEED", getattr(cfg_module, "TRAIN_SEED", 42)))
    experiment = require_attr(cfg_module, "EXPERIMENT_TAG")
    model_root = Path(
        getattr(
            cfg_module,
            "MODEL_ROOT",
            Path(require_attr(cfg_module, "DATASET_DIR")) / "result" / experiment / "model",
        )
    )

    if stage == "train_datapoint_gradient":
        mode = os.environ.get(
            "DATAPOINT_MODEL_MODE",
            os.environ.get(
                "TRAIN_MODE",
                os.environ.get(
                    "ATTRIBUTION_SCORE_MODEL_MODE",
                    os.environ.get(
                        "ATTRIBUTION_SAMPLE_MODEL_MODE",
                        os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo"),
                    ),
                ),
            ),
        )
        mode = canonical_train_model_mode(str(mode))
        return model_root.parent / "model" / mode / f"seed_{train_seed}_train_gradient" / algorithm

    config_values = dict(require_attr(cfg_module, "ATTRIBUTION_CONFIG"))
    if stage == "query_gradient":
        sample_dir = Path(
            os.environ.get(
                "ATTRIBUTION_SAMPLE_DIR",
                str(config_values.get("attribution_sample_dir")),
            )
        )
        sample_seed = int(
            os.environ.get(
                "INITIAL_SEED",
                config_values.get("attribution_sample_seed", getattr(cfg_module, "INITIAL_SEED", 0)),
            )
        )
        return sample_dir / f"seed_{sample_seed:06d}_query_gradient" / algorithm

    score_model_mode = os.environ.get(
        "ATTRIBUTION_SCORE_MODEL_MODE",
        os.environ.get(
            "ATTRIBUTION_SAMPLE_MODEL_MODE",
            os.environ.get("SAMPLE_MODEL_MODE", os.environ.get("TRAIN_MODE", "prompted_solo")),
        ),
    )
    unprompted = (
        os.environ.get("UNPROMPTED", "0") in ("1", "true", "True", "yes")
        or str(score_model_mode).startswith("unprompted_")
    )
    query = os.environ.get("QUERY", getattr(cfg_module, "QUERY", "unconditional"))
    initial_seed = int(os.environ.get("INITIAL_SEED", getattr(cfg_module, "INITIAL_SEED", 0)))
    safe_query = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(query)).strip("_")
    query_component = "unprompted" if unprompted else f"query_{safe_query or 'unconditional'}"
    return (
        model_root.parent
        / "attribution_score"
        / str(score_model_mode)
        / f"train_seed_{train_seed}"
        / query_component
        / f"initial_seed_{initial_seed}"
        / algorithm
        / stage
    )


def stage_root(config_path: str | Path, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage {stage!r}; expected one of {sorted(STAGES)}")
    cfg_module = load_config(config_path)
    algorithm = require_attr(cfg_module, "ALGORITHM")
    return _stage_root(cfg_module, algorithm, stage)


def _write_manifest(out_dir: Path, *, cfg_module: Any, algorithm: str, stage: str, config_values: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "algorithm": algorithm,
        "dataset": require_attr(cfg_module, "DATASET_NAME"),
        "experiment": require_attr(cfg_module, "EXPERIMENT_TAG"),
        "train_seed": int(os.environ.get("TRAIN_SEED", getattr(cfg_module, "TRAIN_SEED", 42))),
        "output_dir": str(out_dir),
        "status": "stage_boundary_ready",
        "note": (
            "This is the pure stage entrypoint. Algorithm kernels should write "
            "only this stage's artifacts here and must not call the monolithic engine."
        ),
        "config": config_values,
    }
    with open(out_dir / "stage_manifest.json", "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def run_stage_config(config_path: str | Path, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage {stage!r}; expected one of {sorted(STAGES)}")

    cfg_module = load_config(config_path)
    algorithm = require_attr(cfg_module, "ALGORITHM")
    config_values = dict(require_attr(cfg_module, "ATTRIBUTION_CONFIG"))
    out_dir = _stage_root(cfg_module, algorithm, stage)
    _write_manifest(
        out_dir,
        cfg_module=cfg_module,
        algorithm=algorithm,
        stage=stage,
        config_values=config_values,
    )

    print("=" * 88)
    print(f"stage     : {stage}")
    print(f"algorithm : {algorithm}")
    print(f"out_dir   : {out_dir}")
    print("=" * 88)
    print("[stage] wrote stage_manifest.json")
    print("[stage] pure compute kernel is intentionally separated from the monolithic engine")
    return out_dir
