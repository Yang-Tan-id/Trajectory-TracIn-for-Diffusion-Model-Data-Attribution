from __future__ import annotations

from dataclasses import asdict
import os
import pickle
from pathlib import Path
from typing import Any, Callable

from .config_loader import load_config, require_attr
from .paths import (
    add_legacy_jax_to_path,
    attribution_run_root,
    chdir_legacy_jax_root,
    ensure_experiment_dirs,
)


ENGINE_MAP: dict[str, tuple[str, str, str]] = {
    "das": (
        "das",
        "EndpointProjectedDASJAXConfig",
        "run_endpoint_das_projected_jax",
    ),
    "traj_tracin": (
        "traj_tracin",
        "TrajAttributionConfig",
        "run_attribution",
    ),
    "dtrak": (
        "dtrak",
        "EndpointDTrakJAXConfig",
        "run_endpoint_dtrak_jax",
    ),
    "end_tracin": (
        "end_tracin",
        "EndpointTraceInConfig",
        "run_endpoint_tracein",
    ),
    "journey_trak": (
        "journey_trak",
        "JourneyTRAKJAXConfig",
        "run_journey_trak_jax",
    ),
}


def build_output_dir(
    dataset_name: str,
    experiment_tag: str,
    algorithm: str,
    query: object,
    initial_seed: int,
) -> str:
    return str(
        (attribution_run_root(dataset_name, experiment_tag, query, initial_seed) / algorithm).resolve()
    )


def _safe_tag(value: object) -> str:
    text = str(value)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    tag = "".join(out).strip("_")
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag


def _damping_tag(value: object) -> str:
    return _safe_tag(str(value).replace("+", "").replace("-", "neg_").replace(".", "p"))


def _range_suffix_from_env() -> str | None:
    text = os.environ.get("ATTRIBUTION_RANGES") or os.environ.get("SCORE_INDEX_RANGES")
    if not text:
        return None
    parts = []
    for part in text.replace(",", " ").split():
        token = part.strip()
        if not token:
            continue
        start_end = token.replace(":", "-").split("-")
        if len(start_end) != 2:
            raise ValueError("ATTRIBUTION_RANGES/SCORE_INDEX_RANGES must look like '1-2500,2501-5000'.")
        parts.append(f"{int(start_end[0])}_{int(start_end[1])}")
    return "range_" + "__".join(parts) if parts else None


def run_algorithm_config(config_path: str | Path) -> Any:
    cfg_module = load_config(config_path)
    dataset_name = require_attr(cfg_module, "DATASET_NAME")
    experiment_tag = require_attr(cfg_module, "EXPERIMENT_TAG")
    algorithm = require_attr(cfg_module, "ALGORITHM")
    config_values = dict(require_attr(cfg_module, "ATTRIBUTION_CONFIG"))

    if algorithm not in ENGINE_MAP:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; expected one of {sorted(ENGINE_MAP)}")

    ensure_experiment_dirs(dataset_name, experiment_tag)
    query = config_values.get("query", os.environ.get("QUERY", "unconditional"))
    initial_seed = int(
        config_values.get("attribution_sample_seed", os.environ.get("INITIAL_SEED", "0"))
    )
    output_algorithm = algorithm
    unprompted = os.environ.get("UNPROMPTED", "0") in ("1", "true", "True", "yes") or str(query) == "unconditional"
    if unprompted:
        output_algorithm = f"{output_algorithm}_unprompted"
    if algorithm == "traj_tracin":
        objective = config_values.get("query_objective", "trajectory_noise_squared_deviation")
        if objective != "trajectory_noise_squared_deviation":
            output_algorithm = f"{algorithm}_{_safe_tag(objective)}"
            if unprompted:
                output_algorithm = f"{output_algorithm}_unprompted"
        parameter_source = str(config_values.get("parameter_source", "ema")).strip().lower()
        if parameter_source not in ("", "ema", "ema_params"):
            output_algorithm = f"{output_algorithm}_{_safe_tag(parameter_source)}"
        range_suffix = _range_suffix_from_env()
        if range_suffix is not None:
            output_algorithm = f"{output_algorithm}_{range_suffix}"
    out_dir = Path(build_output_dir(dataset_name, experiment_tag, output_algorithm, query, initial_seed))
    if algorithm == "das" and os.environ.get("DAS_DAMPING_OUTPUT_TAG"):
        out_dir = out_dir / f"lambda_{_damping_tag(os.environ['DAS_DAMPING_OUTPUT_TAG'])}"
    config_values.setdefault("out_dir", str(out_dir))

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


def run_unprompted_algorithm_config(
    dataset_config_path: str | Path,
    algorithm: str,
) -> Any:
    cfg_module = load_config(dataset_config_path)
    dataset_name = require_attr(cfg_module, "DATASET_NAME")
    experiment_tag = require_attr(cfg_module, "EXPERIMENT_TAG")
    config_factory = require_attr(cfg_module, "unprompted_attribution_config")
    config_values = dict(config_factory(algorithm))

    if algorithm not in ENGINE_MAP:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; expected one of {sorted(ENGINE_MAP)}")

    ensure_experiment_dirs(dataset_name, experiment_tag)
    config_values.setdefault(
        "out_dir",
        build_output_dir(
            dataset_name,
            experiment_tag,
            f"{algorithm}_unprompted",
            config_values.get("query", "unconditional"),
            int(config_values.get("attribution_sample_seed", os.environ.get("INITIAL_SEED", "0"))),
        ),
    )

    add_legacy_jax_to_path()
    chdir_legacy_jax_root()
    module_name, config_class_name, run_name = ENGINE_MAP[algorithm]
    module = __import__(module_name)
    config_class = getattr(module, config_class_name)
    run_fn: Callable[[Any], Any] = getattr(module, run_name)

    reference_ckpt = config_values.get("reference_ckpt")
    if reference_ckpt and Path(reference_ckpt).is_file():
        with open(reference_ckpt, "rb") as handle:
            payload = pickle.load(handle)
        checkpoint_config = payload.get("config", {})
        accepted_fields = set(getattr(config_class, "__dataclass_fields__", {}))
        for key, value in checkpoint_config.items():
            if key in accepted_fields:
                # Do not let training config timesteps=1000 overwrite
                # attribution config timesteps=(0, 200, ..., 999).
                if key == "timesteps":
                    continue
                config_values[key] = value

        if "timesteps_total" in accepted_fields and "timesteps" in checkpoint_config:
            config_values["timesteps_total"] = checkpoint_config["timesteps"]
            
        if checkpoint_config.get("class_cond") is not False:
            raise ValueError(
                f"Expected an unconditional checkpoint, but class_cond is not False: {reference_ckpt}"
            )

    cfg = config_class(**config_values)

    print("=" * 88)
    print(f"dataset       : {dataset_name}")
    print(f"experiment    : {experiment_tag}")
    print(f"algorithm     : {algorithm} (unprompted JAX)")
    print(f"legacy engine : {module_name}.{run_name}")
    print(f"out_dir       : {cfg.out_dir}")
    print("=" * 88)
    result = run_fn(cfg)
    try:
        print("[config]", asdict(cfg))
    except Exception:
        pass
    return result
