from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

try:
    from .config_loader import load_config, require_attr
    from .paths import add_legacy_jax_to_path, chdir_legacy_jax_root, ensure_experiment_dirs
except ImportError:
    import sys

    refine_root = Path(__file__).resolve().parents[1]
    if str(refine_root) not in sys.path:
        sys.path.insert(0, str(refine_root))
    from common.config_loader import load_config, require_attr
    from common.paths import add_legacy_jax_to_path, chdir_legacy_jax_root, ensure_experiment_dirs


def _optional_int(name: str, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _optional_float(name: str, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _checkpoint_dir(cfg_module, algorithm: str) -> str:
    explicit = os.environ.get("JAX_CHECKPOINT_DIR")
    if explicit:
        return explicit
    per_algorithm = os.environ.get("JAX_PER_ALGORITHM", "0") in ("1", "true", "True", "yes")
    if per_algorithm:
        result_root = require_attr(cfg_module, "MODEL_ROOT")
        return str(Path(result_root) / algorithm / "prompted_jax")
    return require_attr(cfg_module, "CHECKPOINT_DIR")


def run_prompted_cifar_training(config_path: str | Path, algorithm: str) -> None:
    cfg_module = load_config(config_path)
    dataset_name = require_attr(cfg_module, "DATASET_NAME")
    experiment_tag = require_attr(cfg_module, "EXPERIMENT_TAG")
    ensure_experiment_dirs(dataset_name, experiment_tag)

    add_legacy_jax_to_path()
    chdir_legacy_jax_root()

    train_mod = __import__("DM__training_CIFAR10_pixel")
    train_cfg = train_mod.TrainConfig(
        data_root=require_attr(cfg_module, "DATA_ROOT"),
        batch_names=None,
        class_names=getattr(cfg_module, "CLASS_NAMES", None),
        use_test=False,
        exclude_ranges=None,
        exclude_indices=None,
        model_type="unet",
        image_size=32,
        in_channels=3,
        base_channels=_optional_int("JAX_BASE_CHANNELS", 160),
        channel_mults=(1, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128,
        num_classes=10,
        class_cond=True,
        cond_mode=getattr(cfg_module, "COMMON_CIFAR", {}).get("cond_mode", "multi_hot"),
        dropout=_optional_float("JAX_DROPOUT", 0.1),
        seed=_optional_int("TRAIN_SEED", 0),
        epochs=_optional_int("JAX_EPOCHS", 200),
        batch_size=_optional_int("JAX_BATCH_SIZE", 256),
        learning_rate=_optional_float("JAX_LEARNING_RATE", 2e-4),
        weight_decay=_optional_float("JAX_WEIGHT_DECAY", 1e-4),
        grad_clip_norm=_optional_float("JAX_GRAD_CLIP_NORM", 1.0),
        ema_decay=_optional_float("JAX_EMA_DECAY", 0.999),
        log_every=_optional_int("JAX_LOG_EVERY", 100),
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        predict_x0=False,
        prefer_device=os.environ.get("JAX_DEVICE", "gpu"),
        use_bfloat16=os.environ.get("JAX_BFLOAT16", "1") in ("1", "true", "True", "yes"),
        use_data_parallel=os.environ.get("JAX_DATA_PARALLEL", "1") in ("1", "true", "True", "yes"),
        num_devices=_optional_int("JAX_NUM_DEVICES", None),
        checkpoint_dir=_checkpoint_dir(cfg_module, algorithm),
        save_every_epochs=_optional_int("JAX_SAVE_EVERY_EPOCHS", 4),
        keep_last_k=_optional_int("JAX_KEEP_LAST_K", None),
        resume_from=os.environ.get("JAX_RESUME_FROM") or None,
        num_workers=_optional_int("JAX_NUM_WORKERS", 0),
        use_tqdm=os.environ.get("JAX_TQDM", "1") in ("1", "true", "True", "yes"),
        prefetch_size=_optional_int("JAX_PREFETCH_SIZE", 4),
        use_wandb=os.environ.get("JAX_USE_WANDB", "0") in ("1", "true", "True", "yes"),
        wandb_project=os.environ.get("JAX_WANDB_PROJECT", "DA-unet-cifar10-pixel-training"),
        wandb_entity=os.environ.get("JAX_WANDB_ENTITY") or "clearoboticslab",
        wandb_run_name=os.environ.get(
            "JAX_WANDB_RUN_NAME",
            f"{dataset_name}-{experiment_tag}-{algorithm}-prompted-jax-seed{_optional_int('TRAIN_SEED', 0)}",
        ),
        wandb_mode=os.environ.get("JAX_WANDB_MODE", "offline"),
        wandb_log_step_metrics=os.environ.get("JAX_WANDB_LOG_STEP_METRICS", "0") in ("1", "true", "True", "yes"),
    )

    print("=" * 88)
    print("Prompted JAX training")
    print(f"dataset       : {dataset_name}")
    print(f"experiment    : {experiment_tag}")
    print(f"algorithm     : {algorithm}")
    print(f"checkpoint_dir: {train_cfg.checkpoint_dir}")
    print(f"class_names   : {train_cfg.class_names}")
    print(f"cond_mode     : {train_cfg.cond_mode}")
    print("=" * 88)
    print(asdict(train_cfg))
    train_mod.train(train_cfg)


def run_prompted_artbench_training(config_path: str | Path, algorithm: str) -> None:
    cfg_module = load_config(config_path)
    dataset_name = require_attr(cfg_module, "DATASET_NAME")
    experiment_tag = require_attr(cfg_module, "EXPERIMENT_TAG")
    ensure_experiment_dirs(dataset_name, experiment_tag)

    add_legacy_jax_to_path()
    chdir_legacy_jax_root()

    train_mod = __import__("DM__training_ARTBENCH_latent")
    model_root = require_attr(cfg_module, "MODEL_ROOT")
    dataset_storage_root = require_attr(cfg_module, "DATASET_STORAGE_ROOT")
    latent_root = require_attr(cfg_module, "LATENT_ROOT")
    train_cfg = train_mod.LatentArtBenchConfig(
        data_root=str(Path(dataset_storage_root) / "raw"),
        train_split="train",
        test_split="test",
        class_names=None,
        image_size=_optional_int("ARTBENCH_IMAGE_SIZE", 256),
        resize_mode=os.environ.get("ARTBENCH_RESIZE_MODE", "shortest_center_crop"),
        file_extensions=(".jpg", ".jpeg", ".png", ".webp"),
        train_exclude_ranges=None,
        train_exclude_indices=None,
        train_exclude_files=None,
        test_exclude_ranges=None,
        test_exclude_indices=None,
        test_exclude_files=None,
        ae_base_channels=_optional_int("ARTBENCH_AE_BASE_CHANNELS", 64),
        ae_epochs=_optional_int("ARTBENCH_AE_EPOCHS", 20),
        ae_batch_size=_optional_int("ARTBENCH_AE_BATCH_SIZE", 32),
        ae_learning_rate=_optional_float("ARTBENCH_AE_LEARNING_RATE", 2e-4),
        ae_weight_decay=_optional_float("ARTBENCH_AE_WEIGHT_DECAY", 1e-4),
        ae_log_every=_optional_int("ARTBENCH_AE_LOG_EVERY", 50),
        latent_channels=_optional_int("ARTBENCH_LATENT_CHANNELS", 4),
        ae_downsample_factor=_optional_int("ARTBENCH_AE_DOWNSAMPLE_FACTOR", 4),
        dm_model_type="unet",
        dm_cond_mode=getattr(cfg_module, "COMMON_ARTBENCH", {}).get("cond_mode", "multi_hot"),
        dm_epochs=_optional_int("JAX_EPOCHS", 100),
        dm_batch_size=_optional_int("JAX_BATCH_SIZE", 128),
        dm_learning_rate=_optional_float("JAX_LEARNING_RATE", 2e-4),
        dm_weight_decay=_optional_float("JAX_WEIGHT_DECAY", 1e-4),
        dm_grad_clip_norm=_optional_float("JAX_GRAD_CLIP_NORM", 1.0),
        dm_ema_decay=_optional_float("JAX_EMA_DECAY", 0.999),
        dm_log_every=_optional_int("JAX_LOG_EVERY", 100),
        dm_timesteps=1000,
        dm_beta_start=1e-4,
        dm_beta_end=0.02,
        dm_predict_x0=False,
        dm_base_channels=_optional_int("JAX_BASE_CHANNELS", 160),
        dm_channel_mults=(1, 2, 2),
        dm_num_res_blocks=2,
        dm_time_emb_dim=128,
        dm_dropout=_optional_float("JAX_DROPOUT", 0.1),
        seed=_optional_int("TRAIN_SEED", 0),
        use_bfloat16=os.environ.get("JAX_BFLOAT16", "1") in ("1", "true", "True", "yes"),
        prefer_device=os.environ.get("JAX_DEVICE", "gpu"),
        use_tqdm=os.environ.get("JAX_TQDM", "1") in ("1", "true", "True", "yes"),
        reuse_autoencoder=os.environ.get("ARTBENCH_REUSE_AUTOENCODER", "1") in ("1", "true", "True", "yes"),
        cache_dir=str(latent_root),
        autoencoder_model_dir=str(Path(model_root) / "autoencoder"),
        dm_checkpoint_dir=_checkpoint_dir(cfg_module, algorithm),
        keep_last_k=_optional_int("JAX_KEEP_LAST_K", None),
    )

    print("=" * 88)
    print("Prompted ArtBench JAX latent training")
    print(f"dataset          : {dataset_name}")
    print(f"experiment       : {experiment_tag}")
    print(f"algorithm        : {algorithm}")
    print(f"dm_checkpoint_dir: {train_cfg.dm_checkpoint_dir}")
    print(f"autoencoder_dir  : {train_cfg.autoencoder_model_dir}")
    print(f"cache_dir        : {train_cfg.cache_dir}")
    print("=" * 88)
    train_mod.train(train_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompted JAX training from a refine dataset config.")
    parser.add_argument("config", type=str)
    parser.add_argument("--algorithm", default=os.environ.get("ALGORITHM", "shared"))
    args = parser.parse_args()

    cfg_module = load_config(args.config)
    dataset_name = require_attr(cfg_module, "DATASET_NAME")
    if dataset_name == "artbench":
        run_prompted_artbench_training(args.config, args.algorithm)
    else:
        run_prompted_cifar_training(args.config, args.algorithm)


if __name__ == "__main__":
    main()
