"""Regression model for recipe rating or quality prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recipe_discovery.utils.io import ensure_parent_dir


def _as_2d_numeric_matrix(x: np.ndarray | Any) -> np.ndarray:
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}.")
    return matrix


def _as_1d_numeric_vector(y: np.ndarray | Any, name: str) -> np.ndarray:
    vector = np.asarray(y, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D {name} vector, got shape {vector.shape}.")
    return vector


class RecipeRegressor:
    """Stable baseline regressor for tabular recipe features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.model: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> RecipeRegressor:
        """Train the regression model."""
        x_matrix = _as_2d_numeric_matrix(x)
        y_vector = _as_1d_numeric_vector(y, "target")
        if x_matrix.shape[0] != y_vector.shape[0]:
            raise ValueError(
                "Feature matrix and target vector must have the same row count: "
                f"x_rows={x_matrix.shape[0]}, y_rows={y_vector.shape[0]}"
            )

        fit_kwargs: dict[str, np.ndarray] = {}
        if sample_weight is not None:
            weights = _as_1d_numeric_vector(sample_weight, "sample_weight")
            if weights.shape[0] != x_matrix.shape[0]:
                raise ValueError(
                    "Sample weight vector must match feature row count: "
                    f"weights={weights.shape[0]}, x_rows={x_matrix.shape[0]}"
                )
            fit_kwargs["ridge__sample_weight"] = weights

        self.model.fit(x_matrix, y_vector, **fit_kwargs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict recipe scores."""
        x_matrix = _as_2d_numeric_matrix(x)
        return np.asarray(self.model.predict(x_matrix), dtype=float)

    def save(self, path: str | Path) -> None:
        """Persist the trained pipeline and metadata."""
        output_path = ensure_parent_dir(path)
        joblib.dump({"alpha": self.alpha, "pipeline": self.model}, output_path)

    @classmethod
    def load(cls, path: str | Path) -> RecipeRegressor:
        """Load a persisted regressor from disk."""
        payload = joblib.load(Path(path))
        if isinstance(payload, dict) and "pipeline" in payload:
            model = cls(alpha=float(payload.get("alpha", 1.0)))
            model.model = payload["pipeline"]
            return model

        # Backward compatibility for older artifacts that stored only estimator.
        model = cls()
        model.model = payload
        return model
