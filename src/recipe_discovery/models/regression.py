"""Regression model for recipe rating or quality prediction."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
import joblib


class RecipeRegressor:
    """Simple baseline regressor."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the regression model."""
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict recipe scores."""
        return self.model.predict(x)

    def save(self, path: str) -> None:
        """Persist trained model."""
        joblib.dump(self.model, path)
