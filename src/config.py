"""Config loader. Keeps every tunable in config/config.yaml so the code stays
declarative and experiments are reproducible."""
from __future__ import annotations

import os

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(_ROOT, "config", "config.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict, *parts: str) -> str:
    """Resolve a path relative to the project root and make sure it exists."""
    full = os.path.join(_ROOT, *parts)
    os.makedirs(os.path.dirname(full) if os.path.splitext(full)[1] else full, exist_ok=True)
    return full
