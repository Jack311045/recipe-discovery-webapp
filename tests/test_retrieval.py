"""Tests for retrieval helpers."""

from __future__ import annotations

import numpy as np

from recipe_discovery.retrieval.similarity import cosine_similarity


def test_cosine_similarity_shape() -> None:
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = cosine_similarity(query, matrix)
    assert scores.shape == (2,)
