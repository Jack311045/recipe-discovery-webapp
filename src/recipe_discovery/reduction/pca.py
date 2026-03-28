"""PCA-based linear dimensionality reduction."""

from __future__ import annotations

import numpy as np


class PCAProjector:
    """Small PCA implementation using SVD."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCAProjector":
        """Fit PCA directions."""
        self.mean_ = x.mean(axis=0, keepdims=True)
        centered = x - self.mean_
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project data to low-dimensional space."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCAProjector is not fitted.")
        centered = x - self.mean_
        return centered @ self.components_.T

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit PCA and return the projected coordinates."""
        self.fit(x)
        return self.transform(x)
