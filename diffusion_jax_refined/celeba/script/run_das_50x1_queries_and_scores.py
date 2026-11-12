#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


CELEBA_ROOT = Path(__file__).resolve().parents[1]
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))

from dataset_config_attr40 import _prompt_path_tag
from dataset_config_das_attr40 import DAS_DAMPING_SWEEP_VALUES


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.replace(",", " ").split() if value]


def run(command: list[str], *, env: dict[str, str], log: Path, execute: bool) -> None:
    print(f"[command] gpu={env.get('CUDA_VISIBLE_DEVICES')} {' '.join(command)}", flush=True)
    if not execute:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(
            command,
            cwd=CELEBA_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CelebA DAS 50x1 query artifacts and score a lambda sweep."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(72)))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--namespace", default="das_50x1_mc1")
    parser.add_argument("--damping-values", default="")
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    gpu_ids = [str(value) for value in parse_ints(args.gpus)]
    if not query_ids or not gpu_ids:
        parser.error("--query-ids and --gpus must not be empty")
    records = json.loads(args.query_file.read_text())["queries"]
    selected: list[tuple[int, str, int]] = []
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        selected.append(
            (
                int(record.get("query_id", query_id)),
                str(record["prompt"]),
                int(record["initial_seed"]),
            )
        )

    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    checkpoint = (
        result_root / "model" / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_0200.ckpt"
    )
    train_artifact = (
        result_root / "model" / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient" / args.namespace
        / "train_datapoint_gradient_artifact.npz"
    )
    if args.execute:
        for required in (args.query_file, checkpoint, train_artifact):
            if not required.is_file():
                raise FileNotFoundError(required)

    log_root = result_root / "logs" / args.namespace
    base_env = os.environ.copy()
    base_env.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS="200",
        DATAPOINT_MODEL_MODE="prompted_solo",
        SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SCORE_MODEL_MODE="prompted_solo",
        SAMPLE_ROOT=str(sample_root),
        DAS_NUM_MC_NOISE="1",
        DAS_PROJ_DIM="4096",
        DAS_DAMPING_SWEEP="1",
        DAS_SHERMAN_MORRISON_DENOMINATOR="0",
        DAS_SCORE_DENOMINATOR_CACHE="0",
        DAS_QUERY_INPUT_MODE="endpoint_renoise",
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
        TF_GPU_ALLOCATOR=os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONUNBUFFERED="1",
    )

    assignments = [selected[index::len(gpu_ids)] for index in range(len(gpu_ids))]

    def query_worker(gpu: str, tasks: list[tuple[int, str, int]]) -> None:
        jobs = []
        for query_id, prompt, seed in tasks:
            prompt_tag = _prompt_path_tag(prompt)
            sample_dir = (
                sample_root / "cifar" / f"prompt_{prompt_tag}"
                / f"model_prompted_solo__ckpt_{checkpoint.stem}"
            )
            artifact = (
                sample_dir / f"seed_{seed:06d}_query_gradient_{args.namespace}"
                / "das" / "query_gradient_artifact.npz"
            )
            if args.skip_query_gradient or artifact.is_file():
                print(f"[skip] DAS query Q{query_id}: {artifact}", flush=True)
                continue
            jobs.append(
                {
                    "label": f"Q{query_id}",
                    "prompt": prompt,
                    "seed": seed,
                    "sample_dir": str(sample_dir),
                    "artifact": str(artifact),
                }
            )
        if not jobs:
            return
        first = jobs[0]
        env = base_env | {
            "CUDA_VISIBLE_DEVICES": gpu,
            "QUERY": str(first["prompt"]),
            "INITIAL_SEED": str(first["seed"]),
            "SAMPLE_SEED": str(first["seed"]),
            "ATTRIBUTION_SAMPLE_DIR": str(first["sample_dir"]),
            "QUERY_GRADIENT_ARTIFACT_PATH": str(first["artifact"]),
            "DAS_QUERY_BATCH_JOBS": json.dumps(jobs),
        }
        run(
            [args.python_bin, "data_attribution/das/02_query_gradient.py"],
            env=env,
            log=log_root / f"query_gpu_{gpu}.log",
            execute=args.execute,
        )

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [
            pool.submit(query_worker, gpu, tasks)
            for gpu, tasks in zip(gpu_ids, assignments, strict=True)
        ]
        for future in futures:
            future.result()

    if args.skip_score:
        return

    jobs = []
    for query_id, prompt, seed in selected:
        prompt_tag = _prompt_path_tag(prompt)
        sample_dir = (
            sample_root / "cifar" / f"prompt_{prompt_tag}"
            / f"model_prompted_solo__ckpt_{checkpoint.stem}"
        )
        artifact = (
            sample_dir / f"seed_{seed:06d}_query_gradient_{args.namespace}"
            / "das" / "query_gradient_artifact.npz"
        )
        if args.execute and not artifact.is_file():
            raise FileNotFoundError(artifact)
        output = (
            result_root / "attribution_score" / "prompted_solo"
            / f"train_seed_{args.train_seed}" / f"query_{prompt_tag}"
            / f"initial_seed_{seed}" / args.namespace / "score"
        )
        jobs.append(
            {"label": f"Q{query_id}", "query_path": str(artifact), "output_dir": str(output)}
        )

    lambdas = (
        parse_floats(args.damping_values)
        if args.damping_values.strip()
        else [float(value) for value in DAS_DAMPING_SWEEP_VALUES]
    )
    lambda_assignments = [lambdas[index::len(gpu_ids)] for index in range(len(gpu_ids))]

    def score_worker(gpu: str, values: list[float]) -> None:
        if not values:
            return
        env = base_env | {
            "CUDA_VISIBLE_DEVICES": gpu,
            "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(train_artifact),
            "DAS_GLOBAL_GRAM_ARTIFACT_PATH": str(train_artifact),
            "DAS_SCORE_BATCH_JOBS": json.dumps(jobs),
            "DAS_SCORE_BACKEND": "jax",
            "DAS_DAMPING_SWEEP_VALUES": ",".join(f"{value:g}" for value in values),
        }
        run(
            [args.python_bin, "data_attribution/das/03_score_batch.py"],
            env=env,
            log=log_root / f"score_gpu_{gpu}.log",
            execute=args.execute,
        )

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [
            pool.submit(score_worker, gpu, values)
            for gpu, values in zip(gpu_ids, lambda_assignments, strict=True)
        ]
        for future in futures:
            future.result()

    print(
        f"[done] queries={len(selected)} lambdas={len(lambdas)} "
        f"timesteps=50 mc=1 points=5000 namespace={args.namespace}",
        flush=True,
    )


if __name__ == "__main__":
    main()

