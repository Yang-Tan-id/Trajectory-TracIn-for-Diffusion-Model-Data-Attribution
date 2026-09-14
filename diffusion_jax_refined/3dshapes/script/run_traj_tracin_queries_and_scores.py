from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def sample_run_root(sample_root: Path, prompt: str, checkpoint: Path) -> Path:
    return (
        sample_root
        / "cifar"
        / f"prompt_{_prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
    )


def query_artifact_path(run_root: Path, seed: int) -> Path:
    return (
        run_root
        / f"seed_{seed:06d}_query_gradient"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, execute: bool) -> None:
    rendered = " ".join(command)
    print(f"[command] gpu={env.get('CUDA_VISIBLE_DEVICES')} {rendered}", flush=True)
    if not execute:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n[command] {rendered}\n")
        log.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3D Shapes DDIM sampling, Traj TracIn query gradients, and four score variants."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    gpu_ids = [str(value) for value in parse_ints(args.gpus)]
    if not query_ids:
        raise ValueError("--query-ids selected no queries")
    if not gpu_ids:
        raise ValueError("--gpus selected no GPUs")

    query_file = SHAPES_ROOT / "queries_seed_0_9.json"
    records = json.loads(query_file.read_text())["queries"]
    selected = []
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        selected.append((query_id, str(record["prompt"]), int(record["initial_seed"])))

    result_root = SHAPES_ROOT / "result" / args.experiment
    checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    train_artifact = (
        result_root
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient"
        / "traj_tracin"
        / "train_datapoint_gradient_artifact.npz"
    )
    sample_root = result_root / "sample_ddim_eta0_1000"
    log_root = result_root / "logs" / "traj_tracin_query_score"
    if args.execute:
        for required in (checkpoint, train_artifact):
            if not required.is_file():
                raise FileNotFoundError(str(required))
        log_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.update(
        PYTHON_BIN=args.python_bin,
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SCORE_MODEL_MODE="prompted_solo",
        TRAJ_QUERY_OBJECTIVE="trajectory_next_checkpoint_noise_mse",
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_NUM_SNAPSHOTS="10",
        TRAJ_TRAIN_MC_SAMPLES="10",
        TRAJ_QUERY_USE_CONFIG_SNAPSHOTS="1",
        TRAJ_TRACIN_PROJ_DIM="4096",
        TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP="1",
        DIFFUSION_TRAJECTORY_SAMPLER="ddim_eta0",
        SAMPLE_TRAJECTORY_STEPS="1000",
        SAVE_TRAJECTORY_PNGS="0",
        SAMPLE_PREFER_DEVICE="gpu",
        SAMPLE_BATCH_SIZE="1",
        SAMPLE_ROOT=str(sample_root),
        TF_GPU_ALLOCATOR=os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
        PYTHONUNBUFFERED="1",
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
    )

    assignments = [selected[index:: len(gpu_ids)] for index in range(len(gpu_ids))]

    def worker(gpu: str, tasks: list[tuple[int, str, int]]) -> None:
        worker_env = base_env | {"CUDA_VISIBLE_DEVICES": gpu}
        log_path = log_root / f"gpu_{gpu}.log"
        for query_id, prompt, seed in tasks:
            run_root = sample_run_root(sample_root, prompt, checkpoint)
            seed_dir = run_root / f"seed_{seed:06d}"
            query_artifact = query_artifact_path(run_root, seed)
            query_env = worker_env | {
                "QUERY": prompt,
                "INITIAL_SEED": str(seed),
                "SAMPLE_SEED": str(seed),
                "SAMPLE_SEEDS": str(seed),
                "ATTRIBUTION_SAMPLE_DIR": str(run_root),
            }
            if args.skip_sampling or (
                (seed_dir / "trajectory_xt.npy").is_file()
                and (seed_dir / "trajectory_t.npy").is_file()
                and (seed_dir / "final_state.npy").is_file()
            ):
                print(f"[skip] query {query_id} complete sample: {seed_dir}", flush=True)
            else:
                run_command(
                    [args.python_bin, "sampling/run_sampling.py"],
                    cwd=SHAPES_ROOT,
                    env=query_env,
                    log_path=log_path,
                    execute=args.execute,
                )
            if args.skip_query_gradient or query_artifact.is_file():
                print(f"[skip] query {query_id} gradient: {query_artifact}", flush=True)
            else:
                run_command(
                    [args.python_bin, "data_attribution/traj_tracin/02_query_gradient.py"],
                    cwd=SHAPES_ROOT,
                    env=query_env,
                    log_path=log_path,
                    execute=args.execute,
                )

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(worker, gpu, tasks) for gpu, tasks in zip(gpu_ids, assignments)]
        for future in futures:
            future.result()

    jobs = []
    for query_id, prompt, seed in selected:
        run_root = sample_run_root(sample_root, prompt, checkpoint)
        query_artifact = query_artifact_path(run_root, seed)
        score_dir = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_tag(prompt)}"
            / f"initial_seed_{seed}"
            / "traj_tracin"
            / "score"
        )
        jobs.append(
            {
                "label": f"query_{query_id}_seed_{seed}",
                "query_path": str(query_artifact),
                "output_dir": str(score_dir),
            }
        )

    if args.skip_score:
        return
    score_env = base_env | {
        "CUDA_VISIBLE_DEVICES": gpu_ids[0],
        "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(train_artifact),
        "TRACIN_SCORE_BATCH_JOBS": json.dumps(jobs),
        "TRACIN_SCORE_QUERY_NORMALIZE": "1",
        "TRACIN_SCORE_TRAIN_NORMALIZE": "1",
        "TRACIN_SCORE_FUSED_BATCH": "1",
        "TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP": "1",
    }
    run_command(
        [args.python_bin, "data_attribution/traj_tracin/04_score_batch.py"],
        cwd=SHAPES_ROOT,
        env=score_env,
        log_path=log_root / "score_batch.log",
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
