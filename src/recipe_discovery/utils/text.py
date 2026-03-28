"""Text normalization helpers."""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def build_recipe_text(title: str, ingredients: str, description: str, tags: str) -> str:
    """Compose a single text field for embedding."""
    parts = [title, ingredients, description, tags]
    return normalize_whitespace(" ".join(p for p in parts if p))
