from __future__ import annotations

"""Train reusable LDS subset models, without running attribution evaluation."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common.config_loader import load_config, require_attr


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _env_int(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return None


def _env_float(*names: str) -> float | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return float(value)
    return None


def _parse_subset_indices(text: str | None, m: int) -> set[int] | None:
    if text in (None, ""):
        return None
    out: set[int] = set()
    for part in text.replace(",", " ").split():
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid subset range {part!r}")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    bad = sorted(x for x in out if x < 0 or x >= m)
    if bad:
        raise ValueError(f"Subset ids out of range for m={m}: {bad[:8]}")
    return out


def _normalize_sample_model_mode(value: str) -> str:
    aliases = {
        "prompt": "prompted_solo",
        "prompted": "prompted_solo",
        "prompted_solo": "prompted_solo",
        "multi": "prompted_multi",
        "prompted_multi": "prompted_multi",
        "unprompted": "unprompted_solo",
        "unprompted_solo": "unprompted_solo",
        "unprompted_multi": "unprompted_multi",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SAMPLE_MODEL_MODE={value!r}; expected one of {', '.join(sorted(aliases))}"
        ) from exc


def _dataset_percentage_to_k(value: float, universe_size: int) -> int:
    if value <= 0:
        raise ValueError("--dataset-percentage must be positive")
    fraction = value if value <= 1 else value / 100.0
    if fraction > 1:
        raise ValueError("--dataset-percentage cannot exceed 100")
    return max(1, min(universe_size, int(round(universe_size * fraction))))


def _percentage_tag(value: float | None) -> str | None:
    if value is None:
        return None
    return f"pct_{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reusable LDS subset models.")
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--m", type=int, default=int(os.environ.get("LDS_M", os.environ.get("LDS_NUM_SUBSETS", "100"))))
    parser.add_argument("--k", type=int, default=_env_int("LDS_K", "LDS_SUBSET_SIZE"))
    parser.add_argument("--dataset-percentage", type=float, default=_env_float("LDS_DATASET_PERCENTAGE", "LDS_DATASET_PERCENT"))
    parser.add_argument(
        "--model-train-seed",
        type=int,
        default=_env_int("LDS_MODEL_TRAIN_SEED", "LDS_TRAIN_SEED", "TRAIN_SEED"),
        help="Seed used by each retrained LDS model.",
    )
    parser.add_argument(
        "--sample-model-mode",
        default=os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo"),
        help="Model family to train LDS subsets for: prompted_solo/prompted_multi/unprompted_solo/unprompted_multi.",
    )
    parser.add_argument(
        "--sample-random-seed",
        type=int,
        default=int(os.environ.get("LDS_SAMPLE_RANDOM_SEED", os.environ.get("LDS_SUBSET_SEED", "0"))),
    )
    parser.add_argument(
        "--subset-indices",
        default=os.environ.get("LDS_SUBSET_INDICES"),
        help="Optional comma/space-separated subset ids or inclusive ranges to train, e.g. '0,4,8' or '0-15'.",
    )
    parser.add_argument("--unprompted", action="store_true", help="Use the unconditional JAX reference model/config.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.m <= 0:
        parser.error("--m must be positive")
    if args.k is not None and args.k <= 0:
        parser.error("--k must be positive")
    if args.k is not None and args.dataset_percentage is not None:
        parser.error("Use either --k or --dataset-percentage, not both")
    try:
        train_subset_ids = _parse_subset_indices(args.subset_indices, args.m)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        sample_model_mode = _normalize_sample_model_mode(args.sample_model_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.unprompted and not sample_model_mode.startswith("unprompted_"):
        parser.error("--unprompted requires SAMPLE_MODEL_MODE to be unprompted_solo or unprompted_multi")
    use_unprompted = args.unprompted or sample_model_mode.startswith("unprompted_")

    dataset_cfg = load_config(args.config)
    legacy_root = Path(require_attr(dataset_cfg, "LEGACY_JAX_ROOT"))
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    train_mod = __import__(getattr(dataset_cfg, "TRAINING_MODULE_NAME", "DM__training_CIFAR10_pixel"))
    TrainConfig = train_mod.TrainConfig
    from DM_counterfactual_retrain_from_attribution import (
        build_filtered_index_to_cifar_row_map,
        load_base_config,
        selected_indices_to_exclude_indices,
    )
    import LDS.DM_cifar_lds as lds_mod

    lds_mod.TrainConfig = TrainConfig
    lds_mod.train = train_mod.train
    run_train_with_optional_logging = lds_mod.run_train_with_optional_logging

    checkpoint_attr = "UNPROMPTED_JAX_REFERENCE_CKPT" if use_unprompted else "REFERENCE_CKPT"
    base_checkpoint = Path(require_attr(dataset_cfg, checkpoint_attr)).resolve()
    train_cfg = load_base_config(str(base_checkpoint))
    train_cfg.resume_from = None
    train_cfg.exclude_ranges = None
    train_cfg.data_root = str(Path(require_attr(dataset_cfg, "DATA_ROOT")).resolve())
    model_train_seed = args.model_train_seed if args.model_train_seed is not None else int(train_cfg.seed)
    train_cfg.seed = model_train_seed

    # LDS training inherits the normal model's saved TrainConfig. Environment
    # overrides are deliberately limited to operational settings.
    if "LDS_EPOCHS" in os.environ:
        train_cfg.epochs = int(os.environ["LDS_EPOCHS"])
    if "LDS_DEVICE" in os.environ:
        train_cfg.prefer_device = os.environ["LDS_DEVICE"]
    if "LDS_NUM_DEVICES" in os.environ:
        train_cfg.num_devices = int(os.environ["LDS_NUM_DEVICES"])
    train_cfg.save_every_epochs = int(os.environ.get("LDS_SAVE_EVERY_EPOCHS", train_cfg.epochs))
    train_cfg.keep_last_k = int(os.environ.get("LDS_KEEP_LAST_K", "1"))

    row_map = build_filtered_index_to_cifar_row_map(
        data_root=train_cfg.data_root,
        batch_names=train_cfg.batch_names,
        class_names=train_cfg.class_names,
    )
    universe = np.arange(len(row_map), dtype=np.int64)
    try:
        k = _dataset_percentage_to_k(args.dataset_percentage, len(universe)) if args.dataset_percentage is not None else args.k
    except ValueError as exc:
        parser.error(str(exc))
    if k is None:
        k = 5000
    if k > len(universe):
        parser.error(f"subset size {k} exceeds dataset size {len(universe)}")

    run_parts = [f"m_{args.m}", f"k_{k}"]
    pct_tag = _percentage_tag(args.dataset_percentage)
    if pct_tag is not None:
        run_parts.append(pct_tag)
    run_parts.append(f"subset_seed_{args.sample_random_seed}")
    run_name = "_".join(run_parts)
    model_root = Path(require_attr(dataset_cfg, "LDS_MODEL_ROOT"))
    out_dir = model_root / sample_model_mode / f"train_seed_{model_train_seed}" / run_name
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.sample_random_seed)
    subsets = []
    universe_set = set(universe.tolist())
    for subset_id in range(args.m):
        subset_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        kept = np.sort(np.random.default_rng(subset_seed).choice(universe, k, replace=False))
        excluded = np.asarray(sorted(universe_set - set(kept.tolist())), dtype=np.int64)
        subset_dir = models_dir / f"subset_{subset_id:04d}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        np.save(subset_dir / "kept_attribution_indices.npy", kept)
        np.save(subset_dir / "excluded_attribution_indices.npy", excluded)
        metadata = {
            "subset_id": subset_id,
            "subset_seed": subset_seed,
            "subset_size": k,
            "dataset_percentage": args.dataset_percentage,
            "subset_dir": str(subset_dir.resolve()),
        }
        _save_json(subset_dir / "subset_metadata.json", metadata)
        subsets.append(metadata)

    payload = {
        "format_version": 1,
        "dataset": require_attr(dataset_cfg, "DATASET_NAME"),
        "experiment": require_attr(dataset_cfg, "EXPERIMENT_TAG"),
        "mode": "unprompted" if use_unprompted else "prompted",
        "sample_model_mode": sample_model_mode,
        "model_train_seed": model_train_seed,
        "m": args.m,
        "k": k,
        "dataset_percentage": args.dataset_percentage,
        "dataset_universe_size": len(universe),
        "sample_random_seed": args.sample_random_seed,
        "base_checkpoint": str(base_checkpoint),
        "train_config_template": asdict(train_cfg),
        "subsets": subsets,
        "trained_subset_indices": sorted(train_subset_ids) if train_subset_ids is not None else None,
        "complete": False,
    }
    _save_json(out_dir / "lds_model_config.json", payload)
    if args.dry_run:
        print(f"Prepared {args.m} LDS subsets in {out_dir} (dry run)")
        return

    for subset in subsets:
        subset_id = int(subset["subset_id"])
        if train_subset_ids is not None and subset_id not in train_subset_ids:
            continue
        subset_dir = Path(subset["subset_dir"])
        cfg = TrainConfig(**asdict(train_cfg))
        excluded = np.load(subset_dir / "excluded_attribution_indices.npy")
        cfg.exclude_indices = selected_indices_to_exclude_indices(excluded, row_map)
        cfg.checkpoint_dir = str(subset_dir)
        cfg.wandb_run_name = f"lds_{run_name}__subset_{subset_id:04d}"
        _save_json(subset_dir / "train_config.json", {"train_config": asdict(cfg)})
        final_ckpt = subset_dir / f"seed_{cfg.seed}_epoch_{int(cfg.epochs):04d}.ckpt"
        if final_ckpt.is_file() and os.environ.get("FORCE_LDS_TRAIN", "0") not in ("1", "true", "True", "yes"):
            print(f"[{subset_id + 1}/{args.m}] skip existing {final_ckpt.name} in {subset_dir.name}", flush=True)
            continue
        print(f"[{subset_id + 1}/{args.m}] training {subset_dir.name}", flush=True)
        run_train_with_optional_logging(
            cfg,
            str(subset_dir / "train.log"),
            quiet=True,
            prefix=f"[{subset_id + 1}/{args.m}] ",
            progress_bar=True,
        )

    if train_subset_ids is None:
        payload["complete"] = True
    else:
        payload["complete"] = all(
            (
                models_dir
                / f"subset_{subset_id:04d}"
                / f"seed_{model_train_seed}_epoch_{int(train_cfg.epochs):04d}.ckpt"
            ).is_file()
            for subset_id in range(args.m)
        )
    _save_json(out_dir / "lds_model_config.json", payload)
    print(f"Saved reusable LDS models to {out_dir}")


if __name__ == "__main__":
    main()
