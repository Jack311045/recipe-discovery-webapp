"""PCA-based linear dimensionality reduction."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from recipe_discovery.reduction.base import BaseReducer
from recipe_discovery.reduction.utils import normalize_embeddings

logger = logging.getLogger(__name__)


class PCAReducer(BaseReducer):
    """PCA implementation using sklearn."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.model = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit PCA directions."""
        normalized = normalize_embeddings(embeddings)
        self.model.fit(normalized)
        self.is_fitted = True
        logger.info(
            "PCA fitted. Explained variance ratio: %s (Total: %.2f%%)",
            self.model.explained_variance_ratio_,
            sum(self.model.explained_variance_ratio_) * 100,
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project data to low-dimensional space."""
        if not self.is_fitted:
            raise RuntimeError("PCAReducer is not fitted.")
        normalized = normalize_embeddings(embeddings)
        return self.model.transform(normalized)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        joblib.dump(self.model, path)
        logger.info("Saved PCAReducer to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a fitted model."""
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info("Loaded PCAReducer from %s", path)
