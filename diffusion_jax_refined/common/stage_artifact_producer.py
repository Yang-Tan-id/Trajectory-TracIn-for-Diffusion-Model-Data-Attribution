from __future__ import annotations

import os
from pathlib import Path

from .algorithm_runner import run_algorithm_config
from .stage_artifact_runner import QUERY_ARTIFACT, TRAIN_ARTIFACT
from .stage_runner import run_stage_config


class _temporary_env:
    def __init__(self, values: dict[str, str]):
        self.values = values
        self.old: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.old[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _missing_compute_error(stage: str, artifact: str, out_dir: Path, algorithm: str) -> RuntimeError:
    return RuntimeError(
        f"{algorithm} {stage} stage reached the artifact producer, but the real compute kernel "
        f"is not wired yet. Expected this stage to write {out_dir / artifact}. "
        "Do not treat stage_manifest.json as a completed gradient artifact."
    )


def run_train_datapoint_gradient_artifact(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    out_dir = run_stage_config(config_path, "train_datapoint_gradient")
    algorithm = config_path.parent.name
    artifact = out_dir / TRAIN_ARTIFACT
    if artifact.is_file():
        print(f"[stage-1] found existing train artifact: {artifact}")
        return out_dir
    if algorithm == "dtrak":
        with _temporary_env(
            {
                "DTRAK_STAGE_MODE": "train",
                "DTRAK_STAGE_ARTIFACT_PATH": str(artifact),
            }
        ):
            run_algorithm_config(config_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"D-TRAK train stage did not produce {artifact}")
        return out_dir
    if algorithm in ("das", "end_tracin", "traj_tracin"):
        mode_env = {
            "das": "DAS_STAGE_MODE",
            "end_tracin": "END_TRACIN_STAGE_MODE",
            "traj_tracin": "TRAJ_TRACIN_STAGE_MODE",
        }[algorithm]
        path_env = {
            "das": "DAS_STAGE_ARTIFACT_PATH",
            "end_tracin": "END_TRACIN_STAGE_ARTIFACT_PATH",
            "traj_tracin": "TRAJ_TRACIN_STAGE_ARTIFACT_PATH",
        }[algorithm]
        with _temporary_env({mode_env: "train", path_env: str(artifact)}):
            run_algorithm_config(config_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"{algorithm} train stage did not produce {artifact}")
        return out_dir
    raise _missing_compute_error("train_datapoint_gradient", TRAIN_ARTIFACT, out_dir, algorithm)


def run_query_gradient_artifact(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    out_dir = run_stage_config(config_path, "query_gradient")
    algorithm = config_path.parent.name
    artifact = out_dir / QUERY_ARTIFACT
    if artifact.is_file():
        print(f"[stage-2] found existing query artifact: {artifact}")
        return out_dir
    if algorithm == "dtrak":
        with _temporary_env(
            {
                "DTRAK_STAGE_MODE": "query",
                "DTRAK_STAGE_ARTIFACT_PATH": str(artifact),
            }
        ):
            run_algorithm_config(config_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"D-TRAK query stage did not produce {artifact}")
        return out_dir
    if algorithm in ("das", "end_tracin", "traj_tracin"):
        mode_env = {
            "das": "DAS_STAGE_MODE",
            "end_tracin": "END_TRACIN_STAGE_MODE",
            "traj_tracin": "TRAJ_TRACIN_STAGE_MODE",
        }[algorithm]
        path_env = {
            "das": "DAS_STAGE_ARTIFACT_PATH",
            "end_tracin": "END_TRACIN_STAGE_ARTIFACT_PATH",
            "traj_tracin": "TRAJ_TRACIN_STAGE_ARTIFACT_PATH",
        }[algorithm]
        with _temporary_env({mode_env: "query", path_env: str(artifact)}):
            run_algorithm_config(config_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"{algorithm} query stage did not produce {artifact}")
        return out_dir
    raise _missing_compute_error("query_gradient", QUERY_ARTIFACT, out_dir, algorithm)
