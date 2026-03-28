"""Classification evaluation."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return simple classification accuracy."""
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())
