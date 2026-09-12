#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


TARGET_FUNCTIONS = "endpoint_counterfactual,traj_counterfactual,simple_loss,noise_trajectory"
TRAJ_VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)
DAS_LAMBDAS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)


def safe_prompt(prompt: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt.replace(",", "__"))
    return re.sub(r"_+", "_", text).strip("_")[:80]


def lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def run(cmd: list[str], *, cwd: Path, env: dict[str, str], execute: bool) -> None:
    print(("RUN " if execute else "DRY ") + " ".join(cmd))
    if execute:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


def model_dirs(root: Path, experiment: str, train_seed: int) -> str:
    return ",".join(
        str(
            root
            / "result"
            / experiment
            / "lds_model"
            / "prompted_solo"
            / f"train_seed_{train_seed}"
            / f"m_64_k_2500_subset_seed_{subset_seed}"
        )
        for subset_seed in (0, 1, 2)
    )


def score_root(root: Path, experiment: str, train_seed: int, prompt: str, seed: int, algorithm: str) -> Path:
    return (
        root
        / "result"
        / experiment
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{safe_prompt(prompt)}"
        / f"initial_seed_{seed}"
        / algorithm
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete 3D Shapes DDPM attribution/LDS experiment.")
    parser.add_argument("--execute", action="store_true", help="Execute; default is a dry-run plan.")
    parser.add_argument("--input", type=Path, default=None, help="Official 3dshapes.h5 (required unless --skip-prepare).")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-lds-train", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = os.environ.get("PYTHON_BIN", "python3")
    env0 = os.environ.copy()
    env0.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        JAX_LR_SCHEDULE="cosine_warmup",
        JAX_LR_WARMUP_RATIO="0.1",
        JAX_SAVE_EVERY_EPOCHS="4",
        LDS_M="64",
        LDS_K="2500",
        LDS_EPOCHS=str(args.epochs),
        LDS_SAVE_EVERY_EPOCHS=str(args.epochs),
        LDS_KEEP_LAST_K="1",
        DIFFUSION_TRAJECTORY_SAMPLER="ddim_eta0",
        LDS_TRAJECTORY_SAMPLER="ddim_eta0",
        DAS_DAMPING_SWEEP="1",
        DAS_SCORE_DENOMINATOR_CACHE="1",
        TRAJ_QUERY_OBJECTIVE="trajectory_next_checkpoint_noise_mse",
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_NUM_SNAPSHOTS="1000",
        TRACIN_SCORE_QUERY_NORMALIZE="1",
        TRACIN_SCORE_TRAIN_NORMALIZE="1",
    )

    if not args.skip_prepare:
        if args.input is None:
            parser.error("--input is required unless --skip-prepare is used")
        run([python, "script/prepare_3dshapes.py", "--input", str(args.input)], cwd=root, env=env0, execute=args.execute)
    run([python, "script/build_queries.py"], cwd=root, env=env0, execute=args.execute)

    if not args.skip_train:
        run([python, "training/run_training.py", "--algorithm=shared"], cwd=root, env=env0, execute=args.execute)

    if not args.skip_lds_train:
        for subset_seed in (0, 1, 2):
            env = env0 | {"LDS_SAMPLE_RANDOM_SEED": str(subset_seed), "SAMPLE_MODEL_MODE": "prompted_solo"}
            run([python, "lds/run_training.py", "--m", "64", "--k", "2500"], cwd=root, env=env, execute=args.execute)

    query_file = root / "queries_seed_0_9.json"
    if query_file.is_file():
        query_records = json.loads(query_file.read_text())["queries"]
    else:
        sys_path = str(root / "script")
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from build_queries import queries
        query_records = queries()

    # Train features/residuals/Gram are query-independent and shared by all ten queries.
    if not args.skip_attribution:
        for algorithm in ("das", "traj_tracin"):
            stage = root / "data_attribution" / algorithm
            run([python, "01_train_datapoint_gradient.py"], cwd=stage, env=env0, execute=args.execute)

    for record in query_records:
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        env = env0 | {
            "QUERY": prompt,
            "INITIAL_SEED": str(seed),
            "SAMPLE_SEEDS": str(seed),
            "SAMPLE_MODEL_MODE": "prompted_solo",
            "ATTRIBUTION_SCORE_MODEL_MODE": "prompted_solo",
            "DATAPOINT_MODEL_MODE": "prompted_solo",
        }
        if not args.skip_sampling:
            run([python, "sampling/run_sampling.py"], cwd=root, env=env, execute=args.execute)
        if not args.skip_attribution:
            for algorithm in ("das", "traj_tracin"):
                stage = root / "data_attribution" / algorithm
                run([python, "02_query_gradient.py"], cwd=stage, env=env, execute=args.execute)
                run([python, "03_score.py"], cwd=stage, env=env, execute=args.execute)
        if args.skip_eval:
            continue

        lds_dirs = model_dirs(root, args.experiment, args.train_seed)
        das_root = score_root(root, args.experiment, args.train_seed, prompt, seed, "das") / "score"
        for damping in DAS_LAMBDAS:
            tag = f"lambda_{lambda_tag(damping)}"
            run(
                [python, "lds/run_eval.py", "--algorithm", f"das_{tag}", "--lds-model-dirs", lds_dirs,
                 "--score-file", str(das_root / tag), "--target-function", TARGET_FUNCTIONS],
                cwd=root, env=env, execute=args.execute,
            )
        traj_root = score_root(root, args.experiment, args.train_seed, prompt, seed, "traj_tracin")
        for variant, directory in TRAJ_VARIANTS:
            run(
                [python, "lds/run_eval.py", "--algorithm", f"traj_tracin_{variant}", "--lds-model-dirs", lds_dirs,
                 "--score-file", str(traj_root / directory), "--target-function", TARGET_FUNCTIONS],
                cwd=root, env=env, execute=args.execute,
            )


if __name__ == "__main__":
    main()
