"""Generic save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if it does not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as JSON."""
    path = ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
