from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


SHAPES_ROOT = Path(__file__).resolve().parents[1]

TARGET_FUNCTIONS = (
    "endpoint_counterfactual",
    "traj_counterfactual",
    "simple_loss",
    "noise_trajectory",
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def lds_model_dirs(experiment: str, train_seed: int) -> list[Path]:
    root = (
        SHAPES_ROOT
        / "result"
        / experiment
        / "lds_model"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
    )
    return [root / f"m_64_k_2500_subset_seed_{seed}" for seed in range(3)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache the four 3D Shapes LDS true-f targets for all queries and models."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    gpu_ids = [str(value) for value in parse_ints(args.gpus)]
    if not query_ids:
        raise ValueError("--query-ids selected no queries")
    if not gpu_ids:
        raise ValueError("--gpus selected no GPUs")

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    selected: list[tuple[int, str, int]] = []
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        selected.append((query_id, str(record["prompt"]), int(record["initial_seed"])))

    model_dirs = lds_model_dirs(args.experiment, args.train_seed)
    if args.execute:
        for model_dir in model_dirs:
            if not (model_dir / "lds_model_config.json").is_file():
                raise FileNotFoundError(str(model_dir / "lds_model_config.json"))

    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_1000"
    log_root = result_root / "logs" / "lds_true_f"
    log_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS="200",
        SAMPLE_MODEL_MODE="prompted_solo",
        SAMPLE_ROOT=str(sample_root),
        DIFFUSION_TRAJECTORY_SAMPLER="ddim_eta0",
        LDS_TRAJECTORY_SAMPLER="ddim_eta0",
        LDS_DEVICE="gpu",
        LDS_SIMPLE_LOSS_NUM_MC=os.environ.get("LDS_SIMPLE_LOSS_NUM_MC", "10"),
        LDS_SIMPLE_LOSS_MC_SEED=os.environ.get("LDS_SIMPLE_LOSS_MC_SEED", "0"),
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
        PYTHONUNBUFFERED="1",
    )
    model_dirs_arg = ",".join(str(path) for path in model_dirs)
    target_arg = ",".join(TARGET_FUNCTIONS)

    def worker(shard_index: int, gpu: str) -> None:
        worker_env = base_env | {"CUDA_VISIBLE_DEVICES": gpu}
        log_path = log_root / f"gpu_{gpu}.log"
        for query_id, prompt, seed in selected:
            command = [
                args.python_bin,
                "lds/run_eval.py",
                "--lds-model-dirs",
                model_dirs_arg,
                "--target-function",
                target_arg,
                "--trajectory-sampler",
                "ddim_eta0",
                "--true-f-only",
                "--model-shard-index",
                str(shard_index),
                "--model-shard-count",
                str(len(gpu_ids)),
            ]
            print(
                f"[command] gpu={gpu} shard={shard_index}/{len(gpu_ids)} query={query_id}",
                flush=True,
            )
            if not args.execute:
                print(" ".join(command), flush=True)
                continue
            query_env = worker_env | {
                "QUERY": prompt,
                "INITIAL_SEED": str(seed),
                "SAMPLE_SEED": str(seed),
            }
            with log_path.open("a") as log:
                log.write(
                    f"\n[query {query_id}] seed={seed} prompt={prompt} "
                    f"shard={shard_index}/{len(gpu_ids)}\n"
                )
                log.flush()
                subprocess.run(
                    command,
                    cwd=SHAPES_ROOT,
                    env=query_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(worker, index, gpu) for index, gpu in enumerate(gpu_ids)]
        for future in futures:
            future.result()

    print(f"All LDS true-f shards complete. Cache root: {result_root / 'eval'}", flush=True)


if __name__ == "__main__":
    main()
