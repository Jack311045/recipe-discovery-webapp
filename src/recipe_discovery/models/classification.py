"""Optional classification module for recipe tags or categories."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class RecipeClassifier:
    """Simple baseline classifier."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier."""
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(x)
