"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configuration files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read dotted keys from a nested dictionary."""
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
