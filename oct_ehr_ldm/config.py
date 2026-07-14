from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProjectConfig:
    """Small JSON configuration wrapper with project-relative path handling."""

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self.data = data
        self.source = source.resolve()
        configured_root = data.get("project_root", "..")
        root = Path(configured_root).expanduser()
        self.root = (self.source.parent / root).resolve() if not root.is_absolute() else root.resolve()

    def get(self, dotted_key: str, default: Any = None) -> Any:
        value: Any = self.data
        for key in dotted_key.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Missing required config value: {dotted_key}")
        return value

    def path(self, dotted_key: str, default: str | None = None) -> Path:
        raw = self.get(dotted_key, default)
        if raw is None:
            raise KeyError(f"Missing required path in config: {dotted_key}")
        path = Path(str(raw)).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must be a JSON object: {source}")
    return ProjectConfig(data, source)
