"""Project settings and configuration helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def get_app_title() -> str:
    """Return the application title."""
    return "Recipe Discovery Web App"
