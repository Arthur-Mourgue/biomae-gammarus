"""Project root resolution and config path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (directory containing src/, configs/, README.md)."""
    env_root = os.environ.get("BIOMAE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # src/biomae/paths.py -> repo root is three levels up
    return Path(__file__).resolve().parents[2]


def config_path(name: str) -> Path:
    """Resolve a YAML config file under configs/."""
    return get_project_root() / "configs" / name


def checkpoint_path(name: str) -> Path:
    """Resolve a file under checkpoints/."""
    return get_project_root() / "checkpoints" / name


def data_path(*parts: str) -> Path:
    """Resolve a path under data/."""
    return get_project_root().joinpath("data", *parts)
