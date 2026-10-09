from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and retain its location for relative paths."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    return config


def resolve_from_config(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the YAML file, not the caller's cwd."""
    value = Path(value)
    if value.is_absolute():
        return value
    return Path(config["_config_path"]).parent.joinpath(value).resolve()
