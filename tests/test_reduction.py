"""Tests for PCA projector."""

from __future__ import annotations

import numpy as np

from recipe_discovery.reduction.pca import PCAReducer


def test_pca_projection_shape() -> None:
    x = np.random.randn(10, 5)
    projector = PCAReducer(n_components=2)
    z = projector.fit_transform(x)
    assert z.shape == (10, 2)
