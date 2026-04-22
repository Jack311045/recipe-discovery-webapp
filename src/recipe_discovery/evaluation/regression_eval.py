"""Regression evaluation."""

from __future__ import annotations

import numpy as np

from recipe_discovery.evaluation.metrics import mean_absolute_error, mean_squared_error


def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a deterministic regression metric summary."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(
            "Regression evaluation requires matching shapes: "
            f"y_true={truth.shape}, y_pred={pred.shape}"
        )

    mse = mean_squared_error(truth, pred)
    rmse = float(np.sqrt(mse))

    residual_sum_squares = float(np.sum((truth - pred) ** 2))
    total_sum_squares = float(np.sum((truth - np.mean(truth)) ** 2))
    if np.isclose(total_sum_squares, 0.0):
        r2 = 1.0 if np.isclose(residual_sum_squares, 0.0) else 0.0
    else:
        r2 = float(1.0 - (residual_sum_squares / total_sum_squares))

    return {
        "mae": mean_absolute_error(truth, pred),
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
