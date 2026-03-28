"""Retrieval index structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InMemoryIndex:
    """Simple in-memory embedding index."""

    embeddings: np.ndarray

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Return top-k indices by cosine similarity."""
        scores = self.embeddings @ query_vector
        ranked = np.argsort(-scores)
        return ranked[:top_k]
