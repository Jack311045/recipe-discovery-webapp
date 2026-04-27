"""Utilities for generating image-search-friendly keywords."""

from __future__ import annotations

import re

STRIP_WORDS = {
    "easy",
    "quick",
    "simple",
    "best",
    "homemade",
    "classic",
    "traditional",
    "minutes",
    "minute",
    "hour",
    "style",
    "recipe",
    "dish",
    "serving",
    "delicious",
    "perfect",
    "amazing",
}


def extract_image_query(recipe_name: str, ingredients: list[str] | None = None) -> str:
    """Return a cleaned image-search query from a recipe name."""
    name = recipe_name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    words = [w for w in name.split() if w and w not in STRIP_WORDS]
    clean = " ".join(words[:5])

    if len(words) < 2 and ingredients:
        clean = f"{' '.join(ingredients[:2])} {clean}".strip()

    return f"{clean} food plated"
