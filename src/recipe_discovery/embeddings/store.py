"""Embedding artifact save / load helpers.

Artifacts are written to ``data/artifacts/`` by default and include:

* ``recipe_embeddings.npy`` — dense embedding matrix
* ``recipe_ids.csv`` — row-aligned recipe IDs
* ``recipe_texts.csv`` — (optional) the canonical text used per recipe
* ``embedding_metadata.json`` — model name, dimensions, row count, etc.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recipe_discovery.settings import ARTIFACTS_DIR
from recipe_discovery.utils.io import ensure_parent_dir, save_json

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Save helpers
# --------------------------------------------------------------------------- #

def save_embeddings(embeddings: np.ndarray, path: Path | None = None) -> Path:
    """Save the embedding matrix as ``.npy``."""
    path = Path(path) if path else ARTIFACTS_DIR / "recipe_embeddings.npy"
    ensure_parent_dir(path)
    np.save(path, embeddings)
    logger.info("Saved embeddings (%s) -> %s", embeddings.shape, path)
    return path


def save_recipe_ids(recipe_ids: pd.Series, path: Path | None = None) -> Path:
    """Save recipe IDs as a single-column CSV aligned with embedding rows."""
    path = Path(path) if path else ARTIFACTS_DIR / "recipe_ids.csv"
    ensure_parent_dir(path)
    pd.DataFrame({"recipe_id": recipe_ids.values}).to_csv(path, index=False)
    logger.info("Saved %d recipe IDs -> %s", len(recipe_ids), path)
    return path


def save_recipe_texts(texts: pd.Series, path: Path | None = None) -> Path:
    """Save the canonical recipe text used for embedding (optional debug artifact)."""
    path = Path(path) if path else ARTIFACTS_DIR / "recipe_texts.csv"
    ensure_parent_dir(path)
    pd.DataFrame({"recipe_text": texts.values}).to_csv(path, index=False)
    logger.info("Saved %d recipe texts -> %s", len(texts), path)
    return path


def save_embedding_metadata(
    *,
    model_name: str,
    embedding_dim: int,
    num_recipes: int,
    normalize: bool,
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> Path:
    """Save a JSON file with embedding run metadata."""
    path = Path(path) if path else ARTIFACTS_DIR / "embedding_metadata.json"
    payload: dict[str, Any] = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_recipes": num_recipes,
        "normalize": normalize,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    save_json(payload, path)
    logger.info("Saved embedding metadata -> %s", path)
    return path


# --------------------------------------------------------------------------- #
# Load helpers
# --------------------------------------------------------------------------- #

def load_embeddings(path: Path | None = None) -> np.ndarray:
    """Load embeddings from a ``.npy`` file."""
    path = Path(path) if path else ARTIFACTS_DIR / "recipe_embeddings.npy"
    arr = np.load(path)
    logger.info("Loaded embeddings (%s) from %s", arr.shape, path)
    return arr


def load_recipe_ids(path: Path | None = None) -> pd.Series:
    """Load the recipe-ID mapping."""
    path = Path(path) if path else ARTIFACTS_DIR / "recipe_ids.csv"
    return pd.read_csv(path)["recipe_id"]
