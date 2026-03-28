"""Generic save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if it does not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as JSON."""
    path = ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
