"""Embedding save/load helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recipe_discovery.utils.io import ensure_parent_dir


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings as a NumPy array."""
    path = ensure_parent_dir(path)
    np.save(path, embeddings)


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from disk."""
    return np.load(path)
