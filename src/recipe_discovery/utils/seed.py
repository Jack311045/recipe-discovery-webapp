"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set common random seeds."""
    random.seed(seed)
    np.random.seed(seed)
