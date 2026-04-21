"""Similarity functions."""

from __future__ import annotations

import numpy as np


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity scores between one query vector and a matrix."""
    query_arr = np.asarray(query)
    matrix_arr = np.asarray(matrix)

    if matrix_arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix_arr.shape}.")

    if query_arr.ndim == 2 and query_arr.shape[0] == 1:
        query_arr = query_arr.reshape(-1)
    if query_arr.ndim != 1:
        raise ValueError(f"Expected 1D query vector, got shape {query_arr.shape}.")

    if matrix_arr.shape[1] != query_arr.shape[0]:
        raise ValueError(
            "Dimension mismatch between query and matrix: "
            f"query_dim={query_arr.shape[0]}, matrix_dim={matrix_arr.shape[1]}"
        )

    query_norm = np.linalg.norm(query_arr) + 1e-12
    matrix_norm = np.linalg.norm(matrix_arr, axis=1) + 1e-12
    return (matrix_arr @ query_arr) / (matrix_norm * query_norm)