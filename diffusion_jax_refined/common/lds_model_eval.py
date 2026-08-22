from __future__ import annotations

"""Evaluate attribution scores against one or more reusable LDS model folders."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from common.config_loader import load_config, require_attr

TARGET_FUNCTION_CHOICES = (
    "noise_trajectory",
    "projected_trajectory",
    "simple_loss",
    "trajectory_state_mse",
    "endpoint_contarfactual",
    "traj_contarfactual",
    "endpoint_counterfactual",
    "traj_counterfactual",
)


def _paths(text: str) -> list[Path]:
    return [Path(part.strip()).expanduser().resolve() for part in text.split(",") if part.strip()]


def _int_list_env(name: str) -> list[int] | None:
    text = os.environ.get(name)
    if text is None or not text.strip():
        return None
    return [int(part) for part in text.replace(",", " ").split()]


def _compact_model_group_name(model_dirs: list[Path], *, max_len: int = 96) -> str:
    names = [path.name for path in model_dirs]
    joined = "__".join(names)
    if len(joined) <= max_len:
        return joined
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    seeds = []
    for name in names:
        if "_seed_" in name:
            try:
                seeds.append(int(name.rsplit("_seed_", 1)[1].split("_", 1)[0]))
            except ValueError:
                pass
    if seeds:
        seed_tag = f"seeds_{min(seeds)}_{max(seeds)}"
    else:
        seed_tag = f"{len(names)}_models"
    return f"{len(names)}_lds_models_{seed_tag}_{digest}"


def _prediction_tag(subset: str, sign: float) -> str:
    if float(sign).is_integer():
        sign_text = str(int(sign))
    else:
        sign_text = f"{sign:g}"
    sign_text = sign_text.replace("-", "m").replace("+", "p").replace(".", "p")
    return f"pred_{subset}_sign_{sign_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more reusable LDS model folders.")
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--lds-model-dirs", required=True, help="Comma-separated result/.../lds_model folders.")
    parser.add_argument("--score-file", default=None, help="One or more comma-separated attribution result folders. Defaults to this dataset/algorithm's attribution output.")
    parser.add_argument("--algorithm", default=os.environ.get("ALGORITHM", "das"))
    parser.add_argument("--unprompted", action="store_true", help="Evaluate unconditional attribution scores/models.")
    parser.add_argument(
        "--prediction-subset",
        choices=["kept", "removed"],
        default=os.environ.get("LDS_PREDICTION_SUBSET", "kept"),
    )
    parser.add_argument(
        "--prediction-sign",
        type=float,
        default=float(os.environ.get("LDS_PREDICTION_SIGN", "-1")),
    )
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default="max")
    parser.add_argument(
        "--target-function",
        choices=TARGET_FUNCTION_CHOICES,
        default="noise_trajectory",
    )
    parser.add_argument("--trajectory-reduction", choices=["mean", "sum", "snapshot_mean"], default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    dataset_cfg = load_config(args.config)
    legacy_root = Path(require_attr(dataset_cfg, "LEGACY_JAX_ROOT"))
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from LDS.DM_cifar_lds import (
        CifarTargetEvaluator,
        build_score_vector,
        combine_attribution_scores,
        infer_attribution_sample_dir,
        infer_prompt,
        infer_score_metadata,
        latest_checkpoint,
        plot_scatter,
        resolve_score_inputs,
        spearman_corr,
        sum_scores,
        write_csv,
        normalize_target_function,
    )
    args.target_function = normalize_target_function(args.target_function)

    model_dirs = _paths(args.lds_model_dirs)
    if not model_dirs:
        parser.error("--lds-model-dirs is empty")
    expected_dataset = require_attr(dataset_cfg, "DATASET_NAME")
    subset_records = []
    configs = []
    for model_dir in model_dirs:
        config_path = model_dir / "lds_model_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing {config_path}")
        cfg = json.loads(config_path.read_text())
        if cfg.get("dataset") != expected_dataset:
            raise ValueError(f"{model_dir} belongs to dataset {cfg.get('dataset')}, not {expected_dataset}")
        expected_mode = "unprompted" if args.unprompted else "prompted"
        if cfg.get("mode", "prompted") != expected_mode:
            raise ValueError(f"{model_dir} is {cfg.get('mode', 'prompted')}, not {expected_mode}")
        configs.append(cfg)
        for subset in cfg["subsets"]:
            subset_dir = model_dir / "models" / f"subset_{int(subset['subset_id']):04d}"
            subset_records.append((model_dir, subset_dir, subset))
    base_checkpoints = {str(Path(cfg["base_checkpoint"]).resolve()) for cfg in configs}
    if len(base_checkpoints) != 1:
        raise ValueError("All selected LDS model folders must share the same base checkpoint")
    base_checkpoint = base_checkpoints.pop()
    diffusion_timesteps = int(configs[0]["train_config_template"].get("timesteps", 1000))

    score_file = args.score_file or os.environ.get("ATTRIBUTION_RESULT_DIRS")
    if score_file is None:
        root_attr = (
            "UNPROMPTED_ATTRIBUTION_RUN_ROOT" if args.unprompted else "ATTRIBUTION_RUN_ROOT"
        )
        attribution_root = Path(require_attr(dataset_cfg, root_attr))
        ranges = os.environ.get("ATTRIBUTION_RANGES") or os.environ.get("SCORE_INDEX_RANGES")
        if args.unprompted:
            algorithm_dir = args.algorithm if "_unprompted" in args.algorithm else f"{args.algorithm}_unprompted"
        else:
            algorithm_dir = args.algorithm
        if args.algorithm.startswith("traj_tracin") and ranges:
            score_file = ",".join(
                str(attribution_root / f"{algorithm_dir}_range_{part.replace(':', '-').replace('-', '_')}")
                for part in ranges.replace(",", " ").split()
            )
        else:
            matches = sorted(
                path for path in attribution_root.glob(f"{algorithm_dir}*") if path.is_dir()
            )
            score_file = ",".join(str(path) for path in matches) or str(
                attribution_root / algorithm_dir
            )
    score_inputs = resolve_score_inputs(score_file)
    indices, scores, sources = combine_attribution_scores(score_inputs, duplicate_policy=args.duplicate_policy)
    score_map = build_score_vector(indices, scores)
    score_meta = infer_score_metadata(score_inputs)
    score_run_config = score_meta.get("score_run_config", {})
    prompt = infer_prompt(score_inputs) or ("unconditional" if args.unprompted else require_attr(dataset_cfg, "QUERY"))
    sample_attr = "UNPROMPTED_ATTRIBUTION_SAMPLE_DIR" if args.unprompted else "ATTRIBUTION_SAMPLE_DIR"
    sample_dir = infer_attribution_sample_dir(score_inputs) or require_attr(dataset_cfg, sample_attr)
    sample_seed = score_run_config.get(
        "attribution_sample_seed",
        getattr(dataset_cfg, "INITIAL_SEED", os.environ.get("INITIAL_SEED", 0)),
    )
    sample_index = int(score_run_config.get("attribution_sample_index", 0))
    reduction = args.trajectory_reduction or "snapshot_mean"
    simple_loss_timesteps = _int_list_env("LDS_SIMPLE_LOSS_TIMESTEPS") or list(
        range(diffusion_timesteps)
    )
    simple_loss_noise_seeds = _int_list_env("LDS_SIMPLE_LOSS_NOISE_SEEDS")

    evaluator = CifarTargetEvaluator(
        code_file=str(legacy_root / "DM__training_CIFAR10_pixel.py"),
        base_checkpoint=base_checkpoint,
        prompt=prompt,
        prefer_device=os.environ.get("LDS_DEVICE", "gpu"),
        data_root=str(Path(require_attr(dataset_cfg, "DATA_ROOT")).resolve()),
        target_function=args.target_function,
        sample_root=str(Path(sample_dir).resolve()),
        sample_seed=None if sample_seed is None else int(sample_seed),
        sample_index=sample_index,
        max_trajectory_steps=None,
        trajectory_reduction=reduction,
        trajectory_projection=score_meta.get("trajectory_projection"),
        simple_loss_timesteps=simple_loss_timesteps,
        simple_loss_noise_seeds=simple_loss_noise_seeds,
        simple_loss_num_mc=int(os.environ.get("LDS_SIMPLE_LOSS_NUM_MC", "10")),
        simple_loss_mc_seed=int(os.environ.get("LDS_SIMPLE_LOSS_MC_SEED", "0")),
    )
    saved_prompt = (
        evaluator.target_meta.get("seed_info", {}).get("prompt")
        or evaluator.target_meta.get("manifest", {}).get("prompt")
    )
    if saved_prompt is not None:
        expected_tokens = sorted(part.strip() for part in str(prompt).split(",") if part.strip())
        saved_tokens = sorted(part.strip() for part in str(saved_prompt).split(",") if part.strip())
        if expected_tokens != saved_tokens:
            raise ValueError(
                "LDS query and saved trajectory prompt do not match: "
                f"query={prompt!r}, trajectory_prompt={saved_prompt!r}, "
                f"trajectory={evaluator.target_meta.get('trajectory_xt_path')}"
            )

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        names = _compact_model_group_name(model_dirs)
        eval_kind = "lds_unprompted" if args.unprompted else "lds"
        eval_root_attr = "UNPROMPTED_EVAL_RUN_ROOT" if args.unprompted else "EVAL_RUN_ROOT"
        out_dir = (
            Path(require_attr(dataset_cfg, eval_root_attr))
            / eval_kind
            / args.algorithm
            / args.target_function
            / _prediction_tag(args.prediction_subset, args.prediction_sign)
            / names
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for global_id, (model_dir, subset_dir, subset) in enumerate(subset_records):
        kept = np.load(subset_dir / "kept_attribution_indices.npy")
        if args.prediction_subset == "kept":
            prediction_indices = kept
        else:
            prediction_indices = np.load(subset_dir / "excluded_attribution_indices.npy")
        prediction = sum_scores(prediction_indices, score_map, args.prediction_sign)
        checkpoint = latest_checkpoint(str(subset_dir))
        true_f, details = evaluator.evaluate(checkpoint)
        row = {
            "subset_id": global_id,
            "subset_seed": int(subset["subset_seed"]),
            "subset_size": int(len(kept)),
            "prediction_subset": args.prediction_subset,
            "prediction_sign": args.prediction_sign,
            "pred_sum_tau": prediction,
            "true_f": true_f,
            "checkpoint": checkpoint,
            "subset_dir": str(subset_dir),
        }
        rows.append(row)
        (out_dir / f"target_{global_id:04d}.json").write_text(
            json.dumps({**row, "target_details": details}, indent=2)
        )
        write_csv(str(out_dir / "lds_results.csv"), rows)
        print(f"[{global_id + 1}/{len(subset_records)}] {model_dir.name}/{subset_dir.name}", flush=True)

    pred = np.asarray([row["pred_sum_tau"] for row in rows], dtype=np.float64)
    true = np.asarray([row["true_f"] for row in rows], dtype=np.float64)
    lds = spearman_corr(pred, true)
    summary = {
        "algorithm": args.algorithm,
        "mode": "unprompted" if args.unprompted else "prompted",
        "lds_model_dirs": [str(path) for path in model_dirs],
        "score_sources": sources,
        "num_models": len(rows),
        "lds_spearman": lds,
        "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
        "target_function": args.target_function,
        "trajectory_reduction": reduction,
        "prediction_subset": args.prediction_subset,
        "prediction_sign": args.prediction_sign,
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
    plot_scatter(str(out_dir / "lds_scatter.png"), pred, true, f"LDS={lds:.4f} ({100.0 * lds:.2f}%)")
    if np.any(scores < 0):
        squared_map = build_score_vector(indices, np.square(scores))
        squared_rows = []
        for row, (_, subset_dir, _) in zip(rows, subset_records):
            prediction_indices = np.load(
                subset_dir
                / (
                    "kept_attribution_indices.npy"
                    if args.prediction_subset == "kept"
                    else "excluded_attribution_indices.npy"
                )
            )
            squared_row = dict(row)
            squared_row["pred_sum_tau"] = sum_scores(
                prediction_indices, squared_map, args.prediction_sign
            )
            squared_rows.append(squared_row)
        squared_pred = np.asarray([row["pred_sum_tau"] for row in squared_rows])
        squared_lds = spearman_corr(squared_pred, true)
        write_csv(str(out_dir / "lds_results_squared_scores.csv"), squared_rows)
        plot_scatter(
            str(out_dir / "lds_scatter_squared_scores.png"),
            squared_pred,
            true,
            f"Squared-score LDS={squared_lds:.4f} ({100.0 * squared_lds:.2f}%)",
            xlabel="Predicted sum of squared attribution scores",
        )
    print(f"Saved LDS evaluation to {out_dir}")


if __name__ == "__main__":
    main()
