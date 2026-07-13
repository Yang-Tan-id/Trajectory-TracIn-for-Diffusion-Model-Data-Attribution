import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DModel


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_unet_from_config(config_path: str | Path) -> UNet2DModel:
    config = UNet2DModel.load_config(str(config_path))
    config["resnet_time_scale_shift"] = config.get("resnet_time_scale_shift", "scale_shift")
    return UNet2DModel.from_config(config)


def save_args(path: str | Path, args: argparse.Namespace) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

