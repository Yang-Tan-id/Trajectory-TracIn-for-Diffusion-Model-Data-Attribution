from __future__ import annotations

"""Cache Shape3D simple-loss LDS targets checkpoint-major.

Each checkpoint is restored exactly once, then evaluated on every selected
query.  This avoids the query-major evaluator's Q x M checkpoint restores.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for path in (SHAPES_ROOT, REFINE_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.lds_model_eval import (  # noqa: E402
    _compact_model_group_name,
    _read_target_cache,
    _write_target_cache,
)
from LDS.DM_cifar_lds import (  # noqa: E402
    CifarTargetEvaluator,
    find_seed_dir,
    latest_checkpoint,
)

import dataset_config as dataset_cfg  # noqa: E402


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def model_dirs(experiment: str, train_seed: int, subset_seeds: list[int]) -> list[Path]:
    root_override = os.environ.get("LDS_MODEL_ROOT_OVERRIDE", "").strip()
    root = (
        Path(root_override)
        if root_override
        else SHAPES_ROOT
        / "result"
        / experiment
        / "lds_model"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
    )
    return [root / f"m_64_k_2500_subset_seed_{seed}" for seed in subset_seeds]


def sample_model_root(
    sample_root: Path, prompt: str, base_checkpoint: Path
) -> Path:
    return (
        sample_root
        / "cifar"
        / f"prompt_{dataset_cfg._prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{base_checkpoint.stem}"
    )


def load_endpoint(sample_dir: Path, seed: int) -> np.ndarray:
    seed_dir = Path(find_seed_dir(str(sample_dir), seed))
    trajectory = np.load(seed_dir / "trajectory_xt.npy", mmap_mode="r")
    if trajectory.ndim != 5 or trajectory.shape[1] < 1:
        raise ValueError(
            f"expected trajectory [K,B,H,W,C], got {trajectory.shape}: {seed_dir}"
        )
    return np.asarray(trajectory[-1:, 0:1], dtype=np.float32).copy()


def cache_matches_probe_contract(
    cached: tuple[float, dict[str, object]] | None,
    timesteps: list[int],
    noise_seeds: list[int],
) -> bool:
    if cached is None:
        return False
    _, details = cached
    return (
        int(details.get("simple_loss_num_mc_terms", -1)) == len(timesteps)
        and bool(details.get("simple_loss_paired_timesteps_noise", False))
        and details.get("simple_loss_probe_construction") == "paired_zip"
        and [int(value) for value in details.get("simple_loss_sampled_timesteps", [])]
        == timesteps
        and [int(value) for value in details.get("simple_loss_noise_key_seeds", [])]
        == noise_seeds
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--subset-seeds", default="0,1,2")
    parser.add_argument("--model-shard-index", type=int, required=True)
    parser.add_argument("--model-shard-count", type=int, required=True)
    parser.add_argument("--simple-loss-timesteps", required=True)
    parser.add_argument("--simple-loss-noise-seeds", required=True)
    parser.add_argument("--simple-loss-mc-seed", type=int, default=0)
    parser.add_argument("--target-cache-namespace", required=True)
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    subset_seeds = parse_ints(args.subset_seeds)
    timesteps = parse_ints(args.simple_loss_timesteps)
    noise_seeds = parse_ints(args.simple_loss_noise_seeds)
    if len(timesteps) != len(noise_seeds):
        raise ValueError("paired timestep/noise lists must have equal length")
    if not 0 <= args.model_shard_index < args.model_shard_count:
        raise ValueError("invalid model shard")

    records = json.loads(args.query_file.read_text())["queries"]
    queries: list[dict[str, object]] = []
    result_root = SHAPES_ROOT / "result" / args.experiment
    base_checkpoint = Path(dataset_cfg.REFERENCE_CKPT).resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    samples = result_root / "sample_ddim_eta0_1000"

    print(f"[prepare] loading {len(query_ids)} query endpoints once", flush=True)
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        sample_dir = sample_model_root(samples, prompt, base_checkpoint)
        queries.append(
            {
                "id": query_id,
                "prompt": prompt,
                "seed": seed,
                "tag": dataset_cfg._prompt_tag(prompt),
                "sample_dir": sample_dir,
                "x0": load_endpoint(sample_dir, seed),
            }
        )

    dirs = model_dirs(args.experiment, args.train_seed, subset_seeds)
    subset_records: list[tuple[int, Path]] = []
    global_id = 0
    for model_dir in dirs:
        config_path = model_dir / "lds_model_config.json"
        config = json.loads(config_path.read_text())
        for subset in config["subsets"]:
            subset_dir = model_dir / "models" / f"subset_{int(subset['subset_id']):04d}"
            if global_id % args.model_shard_count == args.model_shard_index:
                subset_records.append((global_id, subset_dir))
            global_id += 1

    group = _compact_model_group_name(dirs)
    first = queries[0]
    evaluator = CifarTargetEvaluator(
        code_file=str(LEGACY_ROOT / "DM__training_3DSHAPES_pixel.py"),
        base_checkpoint=str(base_checkpoint),
        prompt=str(first["prompt"]),
        prefer_device="gpu",
        data_root=str(Path(dataset_cfg.DATA_ROOT).resolve()),
        target_function="simple_loss",
        sample_root=str(first["sample_dir"]),
        sample_seed=int(first["seed"]),
        sample_index=0,
        max_trajectory_steps=None,
        trajectory_reduction="snapshot_mean",
        trajectory_projection=None,
        simple_loss_timesteps=timesteps,
        simple_loss_noise_seeds=noise_seeds,
        simple_loss_num_mc=len(timesteps),
        simple_loss_mc_seed=args.simple_loss_mc_seed,
        trajectory_sampler="ddim_eta0",
        simple_loss_pair_timesteps_noise=True,
    )

    total_pairs = len(subset_records) * len(queries)
    processed = 0
    print(
        f"[model-major] shard={args.model_shard_index}/{args.model_shard_count} "
        f"checkpoints={len(subset_records)} queries={len(queries)} "
        f"pairs={total_pairs} probes={len(timesteps)} "
        f"chunk={os.environ.get('SIMPLE_LOSS_EVAL_CHUNK_SIZE', 'all')}",
        flush=True,
    )

    for model_position, (model_id, subset_dir) in enumerate(subset_records, 1):
        checkpoint = latest_checkpoint(str(subset_dir))
        missing: list[tuple[dict[str, object], Path]] = []
        for query in queries:
            cache_path = (
                result_root
                / "eval"
                / "prompted_solo"
                / f"query_{query['tag']}"
                / f"initial_seed_{query['seed']}"
                / "lds_target_cache"
                / "ddim_eta0"
                / args.target_cache_namespace
                / group
                / "simple_loss"
                / f"target_{model_id:04d}.json"
            )
            cached = _read_target_cache(
                cache_path,
                checkpoint=checkpoint,
                target_function="simple_loss",
                trajectory_sampler="ddim_eta0",
            )
            if not cache_matches_probe_contract(cached, timesteps, noise_seeds):
                missing.append((query, cache_path))
            else:
                processed += 1

        if not missing:
            print(
                f"[checkpoint {model_position}/{len(subset_records)}] "
                f"id={model_id} cached={len(queries)} computed=0 "
                f"pairs={processed}/{total_pairs}",
                flush=True,
            )
            continue

        target_adapter = evaluator._make_adapter(checkpoint)
        for query, cache_path in missing:
            true_f, details = evaluator.evaluate_simple_loss_with_adapter(
                target_adapter,
                query["x0"],
                str(query["prompt"]),
            )
            details.update(
                model_major=True,
                query_id=int(query["id"]),
                checkpoint_loaded_once_for_queries=True,
            )
            _write_target_cache(
                cache_path,
                checkpoint=checkpoint,
                subset_dir=subset_dir,
                target_function="simple_loss",
                true_f=true_f,
                details=details,
                trajectory_sampler="ddim_eta0",
            )
            processed += 1
            print(
                f"[pair {processed}/{total_pairs}] checkpoint={model_id:04d} "
                f"query=Q{query['id']} true_f={true_f:.8g}",
                flush=True,
            )
        del target_adapter
        gc.collect()

    print(
        f"[done] model-major shard={args.model_shard_index}/"
        f"{args.model_shard_count} pairs={processed}/{total_pairs}",
        flush=True,
    )


if __name__ == "__main__":
    main()
