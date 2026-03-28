"""Clustering diagnostics."""

from __future__ import annotations

import numpy as np


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Return counts per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}
