"""Load YAML config for MotionSense-VLM."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path] = "config/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file (default: config/default.yaml).

    Returns:
        Nested dict of config (e.g. cfg["data"]["root"], cfg["model"]["clip_name"]).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
