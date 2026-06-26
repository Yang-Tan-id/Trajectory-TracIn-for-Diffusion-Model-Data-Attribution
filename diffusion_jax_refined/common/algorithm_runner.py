from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .config_loader import load_config, require_attr
from .paths import (
    add_legacy_jax_to_path,
    attribution_root,
    chdir_legacy_jax_root,
    ensure_experiment_dirs,
)


ENGINE_MAP: dict[str, tuple[str, str, str]] = {
    "das": (
        "DM_dataAttribution_algo_end_das",
        "EndpointProjectedDASJAXConfig",
        "run_endpoint_das_projected_jax",
    ),
    "traj_tracin": (
        "DM_dataAttribution_algo_traj_tracin",
        "TrajAttributionConfig",
        "run_attribution",
    ),
    "dtrak": (
        "DM_dataAttribution_algo_end_dtrak",
        "EndpointDTrakJAXConfig",
        "run_endpoint_dtrak_jax",
    ),
    "end_tracin": (
        "DM_dataAttribution_algo_end_tracin",
        "EndpointTraceInConfig",
        "run_endpoint_tracein",
    ),
    "journey_trak": (
        "DM_dataAttribution_algo_traj_journeytrak",
        "JourneyTRAKJAXConfig",
        "run_journey_trak_jax",
    ),
}


def build_output_dir(dataset_name: str, experiment_tag: str, algorithm: str) -> str:
    return str(attribution_root(dataset_name, experiment_tag, algorithm).resolve())


def run_algorithm_config(config_path: str | Path) -> Any:
    cfg_module = load_config(config_path)
    dataset_name = require_attr(cfg_module, "DATASET_NAME")
    experiment_tag = require_attr(cfg_module, "EXPERIMENT_TAG")
    algorithm = require_attr(cfg_module, "ALGORITHM")
    config_values = dict(require_attr(cfg_module, "ATTRIBUTION_CONFIG"))

    if algorithm not in ENGINE_MAP:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; expected one of {sorted(ENGINE_MAP)}")

    ensure_experiment_dirs(dataset_name, experiment_tag)
    config_values.setdefault("out_dir", build_output_dir(dataset_name, experiment_tag, algorithm))

    add_legacy_jax_to_path()
    chdir_legacy_jax_root()

    module_name, config_class_name, run_name = ENGINE_MAP[algorithm]
    module = __import__(module_name)
    config_class = getattr(module, config_class_name)
    run_fn: Callable[[Any], Any] = getattr(module, run_name)

    cfg = config_class(**config_values)
    print("=" * 88)
    print(f"dataset       : {dataset_name}")
    print(f"experiment    : {experiment_tag}")
    print(f"algorithm     : {algorithm}")
    print(f"legacy engine : {module_name}.{run_name}")
    print(f"out_dir       : {cfg.out_dir}")
    print("=" * 88)
    result = run_fn(cfg)

    try:
        print("[config]", asdict(cfg))
    except Exception:
        pass
    return result

