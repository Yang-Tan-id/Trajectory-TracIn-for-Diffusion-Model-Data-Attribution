from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_config(path: str | Path) -> ModuleType:
    config_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("refine_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_attr(module: ModuleType, name: str):
    if not hasattr(module, name):
        raise AttributeError(f"{module.__file__} must define {name}")
    return getattr(module, name)

