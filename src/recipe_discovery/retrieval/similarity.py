"""Similarity functions."""

from __future__ import annotations

import numpy as np

def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity scores between one query vector and a matrix."""
    query_norm = np.linalg.norm(query) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ query) / (matrix_norm * query_norm)

# Example usage
if __name__ == "__main__":
    # Example input data
    query = np.array([1, 2, 3])
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(query, matrix)
    print(similarity_scores)
