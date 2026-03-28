"""Tests for regression model."""

from __future__ import annotations

import numpy as np

from recipe_discovery.models.regression import RecipeRegressor


def test_regressor_predict_shape() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = RecipeRegressor()
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == y.shape
