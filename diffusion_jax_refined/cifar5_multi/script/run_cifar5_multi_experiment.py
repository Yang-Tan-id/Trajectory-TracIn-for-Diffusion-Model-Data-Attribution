#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from pathlib import Path


LABELS = ("bird", "horse", "automobile", "dog", "cat")
TARGET_FUNCTIONS = (
    "noise_trajectory",
    "endpoint_counterfactual",
    "traj_counterfactual",
    "simple_loss",
)


def query_tag(query: str) -> str:
    return "unprompted" if query == "unprompted" else query.replace(",", "_")


def damping_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def das_damping_values() -> tuple[float, ...]:
    text = os.environ.get("DAS_DAMPING_SWEEP_VALUES")
    if text:
        return tuple(float(part) for part in text.replace(",", " ").split() if part.strip())
    return (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def lds_model_dirs(root: Path, args: argparse.Namespace, mode: str) -> str:
    fraction = args.lds_percentage if args.lds_percentage <= 1 else args.lds_percentage / 100.0
    k = round(args.size * fraction)
    pct_tag = f"pct_{args.lds_percentage:g}".replace(".", "p")
    seeds = [x.strip() for x in args.lds_subset_seeds.split(",") if x.strip()]
    dirs = [
        root
        / "result"
        / args.experiment
        / "lds_model"
        / mode
        / f"train_seed_{args.train_seed}"
        / f"m_{args.lds_m}_k_{k}_{pct_tag}_subset_seed_{seed}"
        for seed in seeds
    ]
    return ",".join(str(path) for path in dirs)


def attribution_score_dirs(root: Path, args: argparse.Namespace, *, mode: str, query: str, algorithm: str) -> list[tuple[str, Path]]:
    query_component = "unprompted" if query == "unprompted" else f"query_{query_tag(query)}"
    base = (
        root
        / "result"
        / args.experiment
        / "attribution_score"
        / mode
        / f"train_seed_{args.train_seed}"
        / query_component
        / f"initial_seed_{args.sample_seeds.split(',')[0]}"
        / algorithm
        / "score"
    )
    if algorithm == "das":
        return [(f"lambda_{damping_tag(v)}", base / f"lambda_{damping_tag(v)}") for v in das_damping_values()]
    return [("default", base)]


def lds_eval_out_dir(
    root: Path,
    args: argparse.Namespace,
    *,
    mode: str,
    query: str,
    algorithm: str,
    score_tag: str,
    target: str,
) -> Path:
    query_component = "unprompted" if query == "unprompted" else f"query_{query_tag(query)}"
    lds_component = "lds_unprompted" if query == "unprompted" else "lds"
    alg_component = algorithm if score_tag == "default" else f"{algorithm}_{score_tag}"
    return (
        root
        / "result"
        / args.experiment
        / "eval"
        / mode
        / query_component
        / f"initial_seed_{args.sample_seeds.split(',')[0]}"
        / lds_component
        / alg_component
        / target
    )


def choose_prompted_queries(seed: int, count: int = 2) -> list[str]:
    import random

    combos = [",".join(c) for c in itertools.combinations(LABELS, 3)]
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:count]


def run(cmd: list[str], env: dict[str, str], *, cwd: Path, execute: bool) -> None:
    printable = " ".join(cmd)
    prefix = "RUN" if execute else "DRY"
    print(f"[{prefix}] {printable}")
    if execute:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("EXPERIMENT_TAG", args.experiment)
    env.setdefault("TRAIN_SEED", str(args.train_seed))
    env.setdefault("JAX_EPOCHS", str(args.epochs))
    env.setdefault("CIFAR5_MULTI_SIZE", str(args.size))
    env.setdefault("CIFAR5_MULTI_DATA_ROOT", str(args.root.parent / "dataset" / "cifar5_multi" / str(args.size)))
    env.setdefault("DAS_PROJ_DIM", "4096")
    env.setdefault("DAS_DAMPING_SWEEP", "1")
    env.setdefault("DTRAK_PROJ_DIM", "4096")
    env.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env.setdefault("PROJECTED_CACHE_DIM", "4096")
    env.setdefault("PROJECTED_DIMS", "4096")
    env.setdefault("TRACIN_USE_SHARED_TRAIN_GRADIENT", "1")
    env.setdefault("LDS_M", str(args.lds_m))
    env.setdefault("LDS_DATASET_PERCENTAGE", str(args.lds_percentage))
    env.setdefault("LDS_EPOCHS", str(args.lds_epochs))
    env.setdefault("LDS_SAVE_EVERY_EPOCHS", str(args.lds_epochs))
    env.setdefault("LDS_KEEP_LAST_K", "1")
    env.setdefault("LDS_NUM_DEVICES", "1")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cifar5_multi end-to-end experiment.")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default prints the plan.")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample-seeds", default="0")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--lds-m", type=int, default=64)
    parser.add_argument("--lds-percentage", type=float, default=25)
    parser.add_argument("--lds-subset-seeds", default="0,1,2")
    parser.add_argument("--lds-epochs", type=int, default=200)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--skip-lds", action="store_true")
    args = parser.parse_args()

    args.root = Path(__file__).resolve().parents[1]
    env0 = base_env(args)
    queries = choose_prompted_queries(args.query_seed, 2)
    all_queries = queries + ["unprompted"]
    print(f"prompted queries: {queries}")
    print("unprompted query: unprompted")

    if not args.skip_generate:
        run(
            [
                os.environ.get("PYTHON_BIN", "python3"),
                str(args.root / "script" / "generate_cifar5_multi.py"),
                "--size",
                str(args.size),
                "--seed",
                str(args.data_seed),
            ],
            env0,
            cwd=args.root,
            execute=args.execute,
        )

    if not args.skip_train:
        run(["bash", "scripts/00_train.sh"], env0, cwd=args.root, execute=args.execute)
        run(["bash", "scripts/00_train_unprompted.sh"], env0, cwd=args.root, execute=args.execute)

    for query in queries:
        env = env0 | {"QUERY": query, "SAMPLE_SEEDS": args.sample_seeds, "SAMPLE_MODEL_MODE": "prompted_solo"}
        run(["bash", "scripts/00_sample.sh"], env, cwd=args.root, execute=args.execute)
    env = env0 | {"SAMPLE_SEEDS": args.sample_seeds, "SAMPLE_MODEL_MODE": "unprompted_solo", "UNPROMPTED": "1"}
    run(["bash", "scripts/00_sample_unprompted.sh"], env, cwd=args.root, execute=args.execute)

    if not args.skip_attribution:
        algorithms = ("das", "traj_tracin", "end_tracin")
        for query in queries:
            env = env0 | {
                "QUERY": query,
                "INITIAL_SEED": args.sample_seeds.split(",")[0],
                "SAMPLE_MODEL_MODE": "prompted_solo",
                "ATTRIBUTION_SCORE_MODEL_MODE": "prompted_solo",
                "DATAPOINT_MODEL_MODE": "prompted_solo",
                "TRACIN_USE_SHARED_TRAIN_GRADIENT": "1",
            }
            for algorithm in algorithms:
                stage_cwd = args.root / "data_attribution" / algorithm
                run([os.environ.get("PYTHON_BIN", "python3"), "02_query_gradient.py"], env, cwd=stage_cwd, execute=args.execute)
                run([os.environ.get("PYTHON_BIN", "python3"), "01_train_datapoint_gradient.py"], env, cwd=stage_cwd, execute=args.execute)
                run([os.environ.get("PYTHON_BIN", "python3"), "03_score.py"], env, cwd=stage_cwd, execute=args.execute)
        env = env0 | {
            "INITIAL_SEED": args.sample_seeds.split(",")[0],
            "SAMPLE_MODEL_MODE": "unprompted_solo",
            "ATTRIBUTION_SCORE_MODEL_MODE": "unprompted_solo",
            "DATAPOINT_MODEL_MODE": "unprompted_solo",
            "UNPROMPTED": "1",
            "TRACIN_USE_SHARED_TRAIN_GRADIENT": "1",
        }
        for algorithm in algorithms:
            stage_cwd = args.root / "data_attribution" / algorithm
            run([os.environ.get("PYTHON_BIN", "python3"), "02_query_gradient.py"], env, cwd=stage_cwd, execute=args.execute)
            run([os.environ.get("PYTHON_BIN", "python3"), "01_train_datapoint_gradient.py"], env, cwd=stage_cwd, execute=args.execute)
            run([os.environ.get("PYTHON_BIN", "python3"), "03_score.py"], env, cwd=stage_cwd, execute=args.execute)

    if not args.skip_lds:
        for subset_seed in [x.strip() for x in args.lds_subset_seeds.split(",") if x.strip()]:
            env = env0 | {"LDS_SAMPLE_RANDOM_SEED": subset_seed, "SAMPLE_MODEL_MODE": "prompted_solo"}
            run(["bash", "scripts/03_lds_training.sh"], env, cwd=args.root, execute=args.execute)
            env = env0 | {"LDS_SAMPLE_RANDOM_SEED": subset_seed, "SAMPLE_MODEL_MODE": "unprompted_solo"}
            run(["bash", "scripts/03_lds_training_unprompted.sh"], env, cwd=args.root, execute=args.execute)

        for query in all_queries:
            for algorithm in ("das", "traj_tracin", "end_tracin"):
                for target in TARGET_FUNCTIONS:
                    if query == "unprompted":
                        mode = "unprompted_solo"
                        env = env0 | {
                            "INITIAL_SEED": args.sample_seeds.split(",")[0],
                            "UNPROMPTED": "1",
                            "SAMPLE_MODEL_MODE": mode,
                        }
                        lds_dirs = lds_model_dirs(args.root, args, mode)
                    else:
                        mode = "prompted_solo"
                        env = env0 | {
                            "QUERY": query,
                            "INITIAL_SEED": args.sample_seeds.split(",")[0],
                            "SAMPLE_MODEL_MODE": mode,
                        }
                        lds_dirs = lds_model_dirs(args.root, args, mode)

                    for score_tag, score_dir in attribution_score_dirs(
                        args.root,
                        args,
                        mode=mode,
                        query=query,
                        algorithm=algorithm,
                    ):
                        cmd = [
                            os.environ.get("PYTHON_BIN", "python3"),
                            "lds/run_eval.py",
                            "--algorithm",
                            algorithm,
                            "--lds-model-dirs",
                            lds_dirs,
                            "--score-file",
                            str(score_dir),
                            "--target-function",
                            target,
                            "--out-dir",
                            str(
                                lds_eval_out_dir(
                                    args.root,
                                    args,
                                    mode=mode,
                                    query=query,
                                    algorithm=algorithm,
                                    score_tag=score_tag,
                                    target=target,
                                )
                            ),
                        ]
                        if query == "unprompted":
                            cmd.insert(2, "--unprompted")
                        run(cmd, env, cwd=args.root, execute=args.execute)


if __name__ == "__main__":
    main()
