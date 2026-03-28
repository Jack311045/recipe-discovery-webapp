"""Shared model inference helpers."""

from __future__ import annotations

import numpy as np


def safe_predict(model, x: np.ndarray) -> np.ndarray:
    """Call a model's predict method with a small compatibility wrapper."""
    return model.predict(x)
