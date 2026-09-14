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

from dataset_config import _prompt_tag


SCHEMES = (
    ("constant_lr_uniform", "constant1", "uniform"),
    ("cosine_lr_ddim_step_squared", "stored_lr", "ddim_step_squared"),
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score existing aligned 3D Shapes Traj TracIn gradients with two weighting schemes."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    gpu_ids = [str(value) for value in parse_ints(args.gpus)]
    if len(gpu_ids) < len(SCHEMES):
        raise ValueError(f"Need at least {len(SCHEMES)} GPU slots, got {gpu_ids}")
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_1000"
    checkpoint = result_root / "model" / "prompted_jax" / f"seed_{args.train_seed}_epoch_0200.ckpt"
    train_artifact = (
        result_root
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient"
        / "traj_tracin"
        / "train_datapoint_gradient_artifact.npz"
    )
    if args.execute and not train_artifact.is_file():
        raise FileNotFoundError(str(train_artifact))

    jobs = []
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
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
            / "traj_tracin"
            / "query_gradient_artifact.npz"
        )
        if args.execute and not query_artifact.is_file():
            raise FileNotFoundError(str(query_artifact))
        jobs.append((query_id, prompt_tag, seed, query_artifact))

    base_env = os.environ.copy()
    base_env.update(
        PYTHONUNBUFFERED="1",
        TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH=str(train_artifact),
        TRACIN_SCORE_QUERY_NORMALIZE="1",
        TRACIN_SCORE_TRAIN_NORMALIZE="1",
        TRACIN_SCORE_FUSED_BATCH="1",
        TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP="1",
        TRACIN_SCORE_TIMESTEPS_TOTAL="1000",
        TRACIN_SCORE_BETA_START="0.0001",
        TRACIN_SCORE_BETA_END="0.02",
    )
    log_root = result_root / "logs" / "traj_tracin_weighted_scores"
    log_root.mkdir(parents=True, exist_ok=True)

    def worker(scheme: tuple[str, str, str], gpu: str) -> None:
        scheme_name, checkpoint_weighting, timestep_weighting = scheme
        batch_jobs = []
        for query_id, prompt_tag, seed, query_artifact in jobs:
            output_dir = (
                result_root
                / "attribution_score"
                / "prompted_solo"
                / f"train_seed_{args.train_seed}"
                / f"query_{prompt_tag}"
                / f"initial_seed_{seed}"
                / f"traj_tracin_{scheme_name}"
                / "score"
            )
            batch_jobs.append(
                {
                    "label": f"query_{query_id}_seed_{seed}",
                    "query_path": str(query_artifact),
                    "output_dir": str(output_dir),
                }
            )
        env = base_env | {
            "CUDA_VISIBLE_DEVICES": gpu,
            "TRACIN_SCORE_CHECKPOINT_WEIGHTING": checkpoint_weighting,
            "TRACIN_SCORE_TIMESTEP_WEIGHTING": timestep_weighting,
            "TRACIN_SCORE_BATCH_JOBS": json.dumps(batch_jobs),
        }
        command = [args.python_bin, "data_attribution/traj_tracin/04_score_batch.py"]
        print(
            f"[scheme] {scheme_name} gpu={gpu} checkpoint={checkpoint_weighting} "
            f"timestep={timestep_weighting}",
            flush=True,
        )
        if not args.execute:
            return
        with (log_root / f"{scheme_name}.log").open("a") as log:
            subprocess.run(
                command,
                cwd=SHAPES_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=len(SCHEMES)) as pool:
        futures = [pool.submit(worker, scheme, gpu_ids[index]) for index, scheme in enumerate(SCHEMES)]
        for future in futures:
            future.result()

    print("Both weighted Traj TracIn score schemes completed.", flush=True)


if __name__ == "__main__":
    main()
