"""Utility functions for dimensionality reduction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from recipe_discovery.reduction.base import BaseReducer

logger = logging.getLogger(__name__)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings to unit sphere before reduction."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-10, a_max=None)


def procrustes_similarity(pca_2d: np.ndarray, ae_2d: np.ndarray) -> float:
    """Align AE projections to PCA via Procrustes and return residual similarity score.
    Score near 1.0 = AE learned same structure as PCA (likely undertrained).
    Score near 0.0 = totally different (likely overtrained or broken).
    Healthy range: 0.3-0.7.
    """
    from scipy.spatial import procrustes
    try:
        _, _, disparity = procrustes(pca_2d, ae_2d)
        return float(1 - disparity)
    except ValueError:
        return 0.0


def load_reducer(
    method: str = "autoencoder", 
    weights_path: str | Path | None = None
) -> "BaseReducer":
    """
    Load fitted PCA or Autoencoder from disk.
    Falls back to PCA if AE weights not found.
    """
    from recipe_discovery.reduction.pca import PCAReducer
    from recipe_discovery.reduction.autoencoder import AutoencoderReducer

    if method == "autoencoder":
        if weights_path and Path(weights_path).exists():
            logger.info("Loading Autoencoder from %s", weights_path)
            reducer = AutoencoderReducer()
            reducer.load_checkpoint(weights_path)
            return reducer
        logger.warning("Autoencoder weights not found at %s, falling back to PCA.", weights_path)
    
    # Fallback or explicit PCA
    pca_path = None
    if weights_path:
        pca_path = Path(weights_path).parent / "pca_projector.pkl"
    
    reducer = PCAReducer()
    if pca_path and pca_path.exists():
        logger.info("Loading PCA from %s", pca_path)
        reducer.load_checkpoint(pca_path)
    else:
        logger.info("Instantiating untrained PCA reducer.")
    
    return reducer
