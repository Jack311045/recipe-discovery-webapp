"""Cosine-similarity retrieval index backed by scikit-learn.

Build, save, and load a ``NearestNeighbors`` index over recipe embeddings so
that downstream retrieval can find the *k* most similar recipes to a query
vector in sub-linear time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from recipe_discovery.settings import ARTIFACTS_DIR
from recipe_discovery.utils.io import ensure_parent_dir

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = ARTIFACTS_DIR / "recipe_index.joblib"


def build_index(embeddings: np.ndarray, *, n_neighbors: int = 10) -> NearestNeighbors:
    """Fit a cosine ``NearestNeighbors`` index on *embeddings*.

    Parameters
    ----------
    embeddings:
        Dense matrix of shape ``(n_recipes, dim)``.
    n_neighbors:
        Default *k* for queries (can be overridden at query time).

    Returns
    -------
    sklearn.neighbors.NearestNeighbors
        Fitted index ready for ``.kneighbors()`` calls.
    """
    logger.info("Building cosine NN index on %d vectors (dim=%d).", *embeddings.shape)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    logger.info("Index built successfully.")
    return nn


def save_index(index: NearestNeighbors, path: Path | None = None) -> Path:
    """Persist a fitted index to disk via joblib."""
    path = Path(path) if path else DEFAULT_INDEX_PATH
    ensure_parent_dir(path)
    joblib.dump(index, path)
    logger.info("Saved retrieval index -> %s", path)
    return path


def load_index(path: Path | None = None) -> NearestNeighbors:
    """Load a previously saved ``NearestNeighbors`` index."""
    path = Path(path) if path else DEFAULT_INDEX_PATH
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    index = joblib.load(path)
    logger.info("Loaded retrieval index from %s", path)
    return index
