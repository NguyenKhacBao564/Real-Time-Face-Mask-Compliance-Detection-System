import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def load_app_config() -> dict[str, Any]:
    return load_yaml(os.getenv("APP_CONFIG", "configs/app.yaml"))

