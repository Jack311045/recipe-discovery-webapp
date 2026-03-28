"""Unified interface for 2D projection methods."""

from __future__ import annotations

import numpy as np

from recipe_discovery.reduction.pca import PCAProjector


def project_with_pca(x: np.ndarray) -> np.ndarray:
    """Project vectors into 2D using PCA."""
    projector = PCAProjector(n_components=2)
    return projector.fit_transform(x)
