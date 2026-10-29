#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
PROMPT_PATH_TAG_MAX_LENGTH = 150
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.predicted_noise_kl import query_probe_seed


def prompt_tag(prompt: str) -> str:
    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return (
        re.sub(r"_+", "_", text).strip("_")[:PROMPT_PATH_TAG_MAX_LENGTH]
        or "empty_prompt"
    )


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def model_dirs(experiment: str, train_seed: int, subset_seeds: list[int]) -> list[Path]:
    root = (
        CELEBA_ROOT / "result" / experiment / "lds_model" / "prompted_solo"
        / f"train_seed_{train_seed}"
    )
    return [root / f"m_64_k_2500_subset_seed_{seed}" for seed in subset_seeds]


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("\n[command] " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample CelebA Attr40 validation-label query100 and cache four LDS targets."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--subset-seeds", default="0,1,2")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-trajectory-steps", type=int, default=50)
    parser.add_argument(
        "--predicted-noise-kl-num-directions", type=int, default=50
    )
    parser.add_argument("--predicted-noise-kl-seed", type=int, default=2026)
    parser.add_argument(
        "--target-functions",
        default="endpoint_counterfactual,traj_counterfactual,simple_loss,noise_trajectory",
    )
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()
    query_ids = parse_ints(args.query_ids)
    gpu_ids = [str(value) for value in parse_ints(args.gpus)]
    subset_seeds = parse_ints(args.subset_seeds)
    if not gpu_ids or not query_ids:
        parser.error("--gpus and --query-ids must not be empty")
    payload = json.loads(args.query_file.read_text())
    all_queries = payload["queries"]
    selected = [all_queries[index] for index in query_ids]
    selected_by_prompt: dict[int, list[dict[str, object]]] = {}
    for query in selected:
        selected_by_prompt.setdefault(int(query["prompt_id"]), []).append(query)

    result_root = CELEBA_ROOT / "result" / args.experiment
    checkpoint = result_root / "model" / "prompted_jax" / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    data_root = REFINE_ROOT / "dataset" / "celeba" / "10000_attr40"
    sample_root = result_root / f"sample_ddim_eta0_{args.num_inference_steps}"
    logs = result_root / "logs" / f"query100_lds_ddim{args.num_inference_steps}"
    lds_dirs = model_dirs(args.experiment, args.train_seed, subset_seeds)
    requested_targets = [
        value
        for value in args.target_functions.replace(",", " ").split()
        if value
    ]
    target_arg = ",".join(requested_targets)
    if set(requested_targets) & {"simple_loss_50x1", "vlb_nll"}:
        cache_namespace = (
            f"forward_noise_k{args.predicted_noise_kl_num_directions}_"
            f"seed{args.predicted_noise_kl_seed}_query_id_batch1_v2"
        )
    elif "predicted_noise_kl" in requested_targets:
        # Preserve the namespace used by the completed predicted-noise run.
        cache_namespace = (
            f"predicted_noise_kl_k{args.predicted_noise_kl_num_directions}_"
            f"seed{args.predicted_noise_kl_seed}_query_id_batch1_v2"
        )
    else:
        cache_namespace = f"steps_{args.num_inference_steps}"
    if args.execute:
        for required in (checkpoint, data_root / "dataset.npz"):
            if not required.is_file():
                raise FileNotFoundError(required)
        for directory in lds_dirs:
            if not (directory / "lds_model_config.json").is_file():
                raise FileNotFoundError(directory / "lds_model_config.json")

    base_env = os.environ.copy()
    base_env.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SCORE_MODEL_MODE="prompted_solo",
        SAMPLE_ROOT=str(sample_root),
        DIFFUSION_TRAJECTORY_SAMPLER="ddim_eta0",
        LDS_TRAJECTORY_SAMPLER="ddim_eta0",
        LDS_TARGET_CACHE_NAMESPACE=cache_namespace,
        LDS_PREDICTED_NOISE_KL_NUM_DIRECTIONS=str(
            args.predicted_noise_kl_num_directions
        ),
        LDS_DEVICE="gpu",
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
        TF_GPU_ALLOCATOR=os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONUNBUFFERED="1",
    )

    prompt_groups = list(selected_by_prompt.values())
    prompt_assignments = [prompt_groups[index::len(gpu_ids)] for index in range(len(gpu_ids))]

    def sample_worker(gpu: str, groups: list[list[dict[str, object]]]) -> None:
        env = base_env | {"CUDA_VISIBLE_DEVICES": gpu}
        for group in groups:
            prompt = str(group[0]["prompt"])
            seeds = ",".join(str(int(item["initial_seed"])) for item in group)
            sample_run_root = (
                sample_root / "cifar" / f"prompt_{prompt_tag(prompt)}"
                / f"model_prompted_solo__ckpt_{checkpoint.stem}"
            )
            if all(
                (sample_run_root / f"seed_{int(item['initial_seed']):06d}" / "trajectory_xt.npy").is_file()
                for item in group
            ):
                print(f"[skip sample] prompt_id={group[0]['prompt_id']} complete", flush=True)
                continue
            command = [
                args.python_bin,
                "DM___data_attribution_sampler.py",
                "--adapter", "cifar",
                "--code-file", "DM__training_CELEBA_pixel.py",
                "--checkpoint", str(checkpoint),
                "--cifar-data-root", str(data_root),
                "--model-tag", "prompted_solo",
                "--prompt", prompt,
                "--seeds", seeds,
                "--batch-size", "1",
                "--prefer-device", "gpu",
                "--outdir", str(sample_root),
                "--trajectory-sampler", "ddim_eta0",
                "--num-inference-steps", str(args.num_inference_steps),
                "--num-trajectory-steps", str(args.num_trajectory_steps),
                "--upscale", "4",
            ]
            print(f"[sample] gpu={gpu} prompt_id={group[0]['prompt_id']} seeds={seeds}", flush=True)
            if args.execute:
                run(command, cwd=LEGACY_ROOT, env=env, log=logs / f"sample_gpu_{gpu}.log")

    if not args.skip_sampling:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(sample_worker, gpu, groups) for gpu, groups in zip(gpu_ids, prompt_assignments)]
            for future in futures:
                future.result()

    model_arg = ",".join(str(path) for path in lds_dirs)

    def eval_worker(shard_index: int, gpu: str) -> None:
        env = base_env | {"CUDA_VISIBLE_DEVICES": gpu}
        for query in selected:
            prompt = str(query["prompt"])
            seed = int(query["initial_seed"])
            qid = int(query["query_id"])
            query_env = env | {
                "QUERY": prompt,
                "INITIAL_SEED": str(seed),
                "SAMPLE_SEED": str(seed),
                # Query-specific directions, shared by all 192 subset models.
                "LDS_PREDICTED_NOISE_KL_MC_SEED": str(
                    query_probe_seed(args.predicted_noise_kl_seed, qid)
                ),
            }
            command = [
                args.python_bin,
                "lds/run_eval_attr40.py",
                "--lds-model-dirs", model_arg,
                "--target-function", target_arg,
                "--trajectory-sampler", "ddim_eta0",
                "--true-f-only",
                "--model-shard-index", str(shard_index),
                "--model-shard-count", str(len(gpu_ids)),
            ]
            print(f"[eval] gpu={gpu} query={query['query_id']} shard={shard_index}/{len(gpu_ids)}", flush=True)
            if args.execute:
                run(command, cwd=CELEBA_ROOT, env=query_env, log=logs / f"eval_gpu_{gpu}.log")

    if not args.skip_eval:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(eval_worker, index, gpu) for index, gpu in enumerate(gpu_ids)]
            for future in futures:
                future.result()

    output_dir = args.output_dir or (
        result_root / f"query100_lds_csv_ddim{args.num_inference_steps}"
    )
    export_command = [
        args.python_bin,
        "script/export_attr40_query100_lds_csv.py",
        "--query-file", str(args.query_file),
        "--result-root", str(result_root),
        "--subset-seeds", ",".join(str(seed) for seed in subset_seeds),
        "--target-functions", target_arg,
        "--query-ids", ",".join(str(value) for value in query_ids),
        "--output-dir", str(output_dir),
    ]
    if args.execute and not args.skip_eval:
        subprocess.run(export_command, cwd=CELEBA_ROOT, env=base_env, check=True)
    else:
        print("[export] " + " ".join(export_command), flush=True)
    print(f"[done] samples={sample_root} csv={output_dir}", flush=True)


if __name__ == "__main__":
    main()
