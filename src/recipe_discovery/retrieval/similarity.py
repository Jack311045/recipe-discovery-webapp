"""Similarity functions."""

from __future__ import annotations

import numpy as np


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity scores between one query vector and a matrix."""
    query_norm = np.linalg.norm(query) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ query) / (matrix_norm * query_norm)
