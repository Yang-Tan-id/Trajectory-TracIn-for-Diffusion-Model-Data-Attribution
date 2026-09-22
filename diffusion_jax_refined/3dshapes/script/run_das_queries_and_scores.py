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
    parser.add_argument("--query-file", type=Path, default=SHAPES_ROOT / "queries_seed_0_9.json")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--artifact-namespace",
        default="",
        help="Optional isolated query/score namespace, for example trajectory_query100x1.",
    )
    parser.add_argument(
        "--train-artifact-namespace",
        default=None,
        help=(
            "Train artifact namespace to reuse. By default it matches --artifact-namespace; "
            "pass an empty value to reuse the standard das train artifact."
        ),
    )
    parser.add_argument(
        "--query-input-mode",
        choices=("endpoint_renoise", "generation_trajectory"),
        default="endpoint_renoise",
        help="Build each query term from a re-noised endpoint or the saved generation x_t.",
    )
    parser.add_argument(
        "--score-output-namespace",
        default="",
        help="Optional output-only namespace while reusing the selected input artifacts.",
    )
    parser.add_argument(
        "--score-contraction",
        choices=("squared", "linear"),
        default="squared",
    )
    parser.add_argument("--timesteps", default="", help="Optional comma-separated DAS timesteps.")
    parser.add_argument("--num-mc-noise", type=int, default=1)
    parser.add_argument(
        "--aggregate-mc-gradient",
        action="store_true",
        help="Average MC gradients within each timestamp and store one term per timestamp.",
    )
    parser.add_argument(
        "--aggregate-mc-normalized",
        action="store_true",
        help=(
            "Within every timestamp, jointly normalize and average the noise-specific "
            "predicted-noise gradients and residuals before DAS scoring."
        ),
    )
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    namespace = args.artifact_namespace.strip().strip("_/")
    das_name = "das" if not namespace else f"das_{namespace}"
    # Query/score variants need isolated output paths, but may still reuse the
    # expensive train/Gram artifact produced by the standard DAS 100x1 run.
    train_namespace = (
        namespace
        if args.train_artifact_namespace is None
        else args.train_artifact_namespace.strip().strip("_/")
    )
    train_das_name = "das" if not train_namespace else f"das_{train_namespace}"
    score_output_namespace = args.score_output_namespace.strip().strip("_/")
    score_das_name = (
        das_name if not score_output_namespace else f"das_{score_output_namespace}"
    )
    if args.score_contraction != "squared" and score_das_name == das_name:
        raise ValueError(
            "a non-squared DAS score requires --score-output-namespace to avoid overwriting existing scores"
        )
    query_suffix = "query_gradient" if not namespace else f"query_gradient_{namespace}"
    if args.num_mc_noise <= 0:
        raise ValueError("--num-mc-noise must be positive")
    if args.aggregate_mc_gradient and args.aggregate_mc_normalized:
        raise ValueError("the two MC aggregation modes are mutually exclusive")

    query_ids = parse_ints(args.query_ids)
    gpu_ids = [str(value) for value in parse_ints(args.gpus)]
    if not query_ids:
        raise ValueError("--query-ids selected no queries")
    if len(gpu_ids) != 2:
        raise ValueError(f"DAS RTX runner requires exactly two GPUs, got {gpu_ids}")

    records = json.loads(args.query_file.read_text())["queries"]
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_1000"
    checkpoint = result_root / "model" / "prompted_jax" / f"seed_{args.train_seed}_epoch_0200.ckpt"
    train_artifact = (
        result_root
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient"
        / train_das_name
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

    log_tag = score_output_namespace or namespace
    log_root = result_root / "logs" / ("das_query_score" if not log_tag else f"das_query_score_{log_tag}")
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
        DAS_NUM_MC_NOISE=str(args.num_mc_noise),
        DAS_PROJ_DIM="4096",
        DAS_DAMPING_SWEEP="1",
        DAS_SCORE_DENOMINATOR_CACHE="1",
        DAS_SHERMAN_MORRISON_DENOMINATOR="1",
        DAS_SCORE_CONTRACTION=args.score_contraction,
        DAS_QUERY_INPUT_MODE=args.query_input_mode,
        TF_GPU_ALLOCATOR=os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
        PYTHONUNBUFFERED="1",
    )
    if args.timesteps.strip():
        base_env["DAS_TIMESTEPS"] = args.timesteps
    if args.aggregate_mc_gradient:
        base_env["DAS_AGGREGATE_MC_GRADIENT"] = "1"
    if args.aggregate_mc_normalized:
        base_env["DAS_AGGREGATE_MC_NORMALIZED"] = "1"

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
                / f"seed_{seed:06d}_{query_suffix}"
                / das_name
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
                "QUERY_GRADIENT_ARTIFACT_PATH": str(query_artifact),
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
        query_artifact = (
            run_root / f"seed_{seed:06d}_{query_suffix}" / das_name / "query_gradient_artifact.npz"
        )
        if args.execute and not query_artifact.is_file():
            raise FileNotFoundError(str(query_artifact))
        output_dir = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{prompt_tag}"
            / f"initial_seed_{seed}"
            / score_das_name
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

    print(
        f"DAS query gradients and all 16 lambda scores completed | "
        f"query_input={args.query_input_mode} train={train_das_name} "
        f"contraction={args.score_contraction} output={score_das_name}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
