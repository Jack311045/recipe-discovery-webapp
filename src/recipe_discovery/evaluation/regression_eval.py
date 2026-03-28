"""Regression evaluation."""

from __future__ import annotations

import numpy as np

from recipe_discovery.evaluation.metrics import mean_absolute_error, mean_squared_error


def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a minimal regression metric summary."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }
