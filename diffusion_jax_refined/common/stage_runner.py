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
        mode = os.environ.get("DATAPOINT_MODEL_MODE", os.environ.get("TRAIN_MODE", "prompted_solo"))
        return model_root.parent / "model" / mode / f"seed_{train_seed}_train_gradient" / algorithm

    query = os.environ.get("QUERY", getattr(cfg_module, "QUERY", "unconditional"))
    initial_seed = int(os.environ.get("INITIAL_SEED", getattr(cfg_module, "INITIAL_SEED", 0)))
    safe_query = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(query)).strip("_")
    return (
        model_root.parent
        / "attribution_score"
        / f"query_{safe_query or 'unconditional'}"
        / f"initial_seed_{initial_seed}"
        / algorithm
        / stage
    )


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
