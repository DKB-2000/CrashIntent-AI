"""Portable project path configuration loaded from Java-style properties files."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROPERTIES = DEFAULT_ROOT / "project.properties"
LOCAL_PROPERTIES = DEFAULT_ROOT / "project.local.properties"
ENV_PROPERTIES = "CRASHVIDEO_PROJECT_PROPERTIES"


def _read_properties(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        separator = "=" if "=" in line else ":" if ":" in line else None
        if separator is None:
            raise ValueError(f"invalid properties line {path}:{line_number}: {raw_line!r}")
        key, value = line.split(separator, 1)
        values[key.strip()] = value.strip()
    return values


def load_project_properties(properties_path: Path | None = None) -> dict[str, str]:
    """Load tracked defaults, optional local overrides, or one explicit config.

    ``CRASHVIDEO_PROJECT_PROPERTIES`` has the same effect as ``properties_path``.
    An explicit file replaces the default/local pair, which makes CI and Kaggle
    configuration deterministic.
    """
    explicit = properties_path
    if explicit is None and os.getenv(ENV_PROPERTIES):
        explicit = Path(os.environ[ENV_PROPERTIES])
    if explicit is not None:
        explicit = Path(explicit).expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        values = _read_properties(explicit)
        base = explicit.parent
    else:
        values = _read_properties(DEFAULT_PROPERTIES)
        values.update(_read_properties(LOCAL_PROPERTIES))
        base = DEFAULT_ROOT
    root_value = values.get("project.root", ".")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (base / root).resolve()
    values["project.root"] = str(root)
    return values


def configured_path(key: str, properties: dict[str, str] | None = None) -> Path:
    values = properties or load_project_properties()
    if key not in values:
        raise KeyError(f"missing project property: {key}")
    value = Path(values[key]).expanduser()
    if not value.is_absolute():
        value = Path(values["project.root"]) / value
    return value.resolve()
