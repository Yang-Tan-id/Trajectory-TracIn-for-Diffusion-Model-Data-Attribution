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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reusable LDS subset models.")
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--m", type=int, default=int(os.environ.get("LDS_M", "100")))
    parser.add_argument("--k", type=int, default=int(os.environ.get("LDS_K", os.environ.get("LDS_SUBSET_SIZE", "5000"))))
    parser.add_argument(
        "--sample-random-seed",
        type=int,
        default=int(os.environ.get("LDS_SAMPLE_RANDOM_SEED", os.environ.get("LDS_SUBSET_SEED", "0"))),
    )
    parser.add_argument("--unprompted", action="store_true", help="Use the unconditional JAX reference model/config.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.m <= 0 or args.k <= 0:
        parser.error("--m and --k must be positive")

    dataset_cfg = load_config(args.config)
    legacy_root = Path(require_attr(dataset_cfg, "LEGACY_JAX_ROOT"))
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    from DM__training_CIFAR10_pixel import TrainConfig
    from DM_counterfactual_retrain_from_attribution import (
        build_filtered_index_to_cifar_row_map,
        load_base_config,
        selected_indices_to_exclude_indices,
    )
    from LDS.DM_cifar_lds import run_train_with_optional_logging

    checkpoint_attr = "UNPROMPTED_JAX_REFERENCE_CKPT" if args.unprompted else "REFERENCE_CKPT"
    base_checkpoint = Path(require_attr(dataset_cfg, checkpoint_attr)).resolve()
    train_cfg = load_base_config(str(base_checkpoint))
    train_cfg.resume_from = None
    train_cfg.exclude_ranges = None
    train_cfg.data_root = str(Path(require_attr(dataset_cfg, "DATA_ROOT")).resolve())

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
    if args.k > len(universe):
        parser.error(f"--k {args.k} exceeds dataset size {len(universe)}")

    run_name = f"m_{args.m}_k_{args.k}_seed_{args.sample_random_seed}"
    model_root = Path(require_attr(dataset_cfg, "LDS_MODEL_ROOT"))
    out_dir = model_root / "unprompted" / run_name if args.unprompted else model_root / run_name
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.sample_random_seed)
    subsets = []
    universe_set = set(universe.tolist())
    for subset_id in range(args.m):
        subset_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        kept = np.sort(np.random.default_rng(subset_seed).choice(universe, args.k, replace=False))
        excluded = np.asarray(sorted(universe_set - set(kept.tolist())), dtype=np.int64)
        subset_dir = models_dir / f"subset_{subset_id:04d}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        np.save(subset_dir / "kept_attribution_indices.npy", kept)
        np.save(subset_dir / "excluded_attribution_indices.npy", excluded)
        metadata = {
            "subset_id": subset_id,
            "subset_seed": subset_seed,
            "subset_size": args.k,
            "subset_dir": str(subset_dir.resolve()),
        }
        _save_json(subset_dir / "subset_metadata.json", metadata)
        subsets.append(metadata)

    payload = {
        "format_version": 1,
        "dataset": require_attr(dataset_cfg, "DATASET_NAME"),
        "experiment": require_attr(dataset_cfg, "EXPERIMENT_TAG"),
        "mode": "unprompted" if args.unprompted else "prompted",
        "m": args.m,
        "k": args.k,
        "sample_random_seed": args.sample_random_seed,
        "base_checkpoint": str(base_checkpoint),
        "train_config_template": asdict(train_cfg),
        "subsets": subsets,
        "complete": False,
    }
    _save_json(out_dir / "lds_model_config.json", payload)
    if args.dry_run:
        print(f"Prepared {args.m} LDS subsets in {out_dir} (dry run)")
        return

    for subset in subsets:
        subset_id = int(subset["subset_id"])
        subset_dir = Path(subset["subset_dir"])
        cfg = TrainConfig(**asdict(train_cfg))
        excluded = np.load(subset_dir / "excluded_attribution_indices.npy")
        cfg.exclude_indices = selected_indices_to_exclude_indices(excluded, row_map)
        cfg.checkpoint_dir = str(subset_dir)
        cfg.wandb_run_name = f"lds_{run_name}__subset_{subset_id:04d}"
        _save_json(subset_dir / "train_config.json", {"train_config": asdict(cfg)})
        print(f"[{subset_id + 1}/{args.m}] training {subset_dir.name}", flush=True)
        run_train_with_optional_logging(
            cfg,
            str(subset_dir / "train.log"),
            quiet=True,
            prefix=f"[{subset_id + 1}/{args.m}] ",
            progress_bar=True,
        )

    payload["complete"] = True
    _save_json(out_dir / "lds_model_config.json", payload)
    print(f"Saved reusable LDS models to {out_dir}")


if __name__ == "__main__":
    main()
