from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import DAS_DAMPING_SWEEP_VALUES, _prompt_tag


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def run_command(command: list[str], *, env: dict[str, str], log_path: Path, execute: bool) -> None:
    print(f"[command] gpu={env.get('CUDA_VISIBLE_DEVICES')} {' '.join(command)}", flush=True)
    if not execute:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.run(
            command,
            cwd=SHAPES_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3D Shapes DAS query gradients and a two-GPU lambda-sweep score."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    gpu_ids = [str(value) for value in parse_ints(args.gpus)]
    if not query_ids:
        raise ValueError("--query-ids selected no queries")
    if len(gpu_ids) != 2:
        raise ValueError(f"DAS RTX runner requires exactly two GPUs, got {gpu_ids}")

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_1000"
    checkpoint = result_root / "model" / "prompted_jax" / f"seed_{args.train_seed}_epoch_0200.ckpt"
    train_artifact = (
        result_root
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient"
        / "das"
        / "train_datapoint_gradient_artifact.npz"
    )
    selected = []
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        selected.append((query_id, str(record["prompt"]), int(record["initial_seed"])))

    if args.execute:
        for required in (checkpoint, train_artifact):
            if not required.is_file():
                raise FileNotFoundError(str(required))

    log_root = result_root / "logs" / "das_query_score"
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
        DAS_SCORE_DENOMINATOR_CACHE="1",
        DAS_SHERMAN_MORRISON_DENOMINATOR="1",
        TF_GPU_ALLOCATOR=os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
        PYTHONUNBUFFERED="1",
    )

    assignments = [selected[index::2] for index in range(2)]

    def query_worker(gpu: str, tasks: list[tuple[int, str, int]]) -> None:
        for query_id, prompt, seed in tasks:
            prompt_tag = _prompt_tag(prompt)
            run_root = (
                sample_root
                / "cifar"
                / f"prompt_{prompt_tag}"
                / f"model_prompted_solo__ckpt_{checkpoint.stem}"
            )
            query_artifact = (
                run_root
                / f"seed_{seed:06d}_query_gradient"
                / "das"
                / "query_gradient_artifact.npz"
            )
            if args.skip_query_gradient or query_artifact.is_file():
                print(f"[skip] DAS query {query_id}: {query_artifact}", flush=True)
                continue
            env = base_env | {
                "CUDA_VISIBLE_DEVICES": gpu,
                "QUERY": prompt,
                "INITIAL_SEED": str(seed),
                "SAMPLE_SEED": str(seed),
                "ATTRIBUTION_SAMPLE_DIR": str(run_root),
            }
            run_command(
                [args.python_bin, "data_attribution/das/02_query_gradient.py"],
                env=env,
                log_path=log_root / f"gpu_{gpu}_query.log",
                execute=args.execute,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(query_worker, gpu, tasks) for gpu, tasks in zip(gpu_ids, assignments)]
        for future in futures:
            future.result()

    if args.skip_score:
        return

    batch_jobs = []
    for query_id, prompt, seed in selected:
        prompt_tag = _prompt_tag(prompt)
        run_root = (
            sample_root
            / "cifar"
            / f"prompt_{prompt_tag}"
            / f"model_prompted_solo__ckpt_{checkpoint.stem}"
        )
        query_artifact = run_root / f"seed_{seed:06d}_query_gradient" / "das" / "query_gradient_artifact.npz"
        if args.execute and not query_artifact.is_file():
            raise FileNotFoundError(str(query_artifact))
        output_dir = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{prompt_tag}"
            / f"initial_seed_{seed}"
            / "das"
            / "score"
        )
        batch_jobs.append(
            {"label": f"query_{query_id}_seed_{seed}", "query_path": str(query_artifact), "output_dir": str(output_dir)}
        )

    lambdas = [float(value) for value in DAS_DAMPING_SWEEP_VALUES]
    lambda_shards = [lambdas[index::2] for index in range(2)]

    def score_worker(gpu: str, damping_values: list[float]) -> None:
        env = base_env | {
            "CUDA_VISIBLE_DEVICES": gpu,
            "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(train_artifact),
            "DAS_GLOBAL_GRAM_ARTIFACT_PATH": str(train_artifact),
            "DAS_SCORE_BATCH_JOBS": json.dumps(batch_jobs),
            "DAS_SCORE_BACKEND": "jax",
            "DAS_DAMPING_SWEEP_VALUES": ",".join(f"{value:g}" for value in damping_values),
        }
        run_command(
            [args.python_bin, "data_attribution/das/03_score_batch.py"],
            env=env,
            log_path=log_root / f"gpu_{gpu}_score.log",
            execute=args.execute,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(score_worker, gpu, damping_values)
            for gpu, damping_values in zip(gpu_ids, lambda_shards)
        ]
        for future in futures:
            future.result()

    print("DAS query gradients and all 16 lambda scores completed.", flush=True)


if __name__ == "__main__":
    main()
