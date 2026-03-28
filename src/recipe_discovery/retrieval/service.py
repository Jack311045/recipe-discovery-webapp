"""End-to-end retrieval service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recipe_discovery.retrieval.filters import apply_basic_filters
from recipe_discovery.retrieval.similarity import cosine_similarity


@dataclass
class RetrievalRequest:
    """Search request payload."""

    query: str
    top_k: int = 10
    dietary_filter: str | None = None
    max_time_minutes: int | None = None


class RetrievalService:
    """Semantic recipe search service."""

    def __init__(self) -> None:
        self.encoder = None
        self.embeddings: np.ndarray | None = None
        self.metadata: pd.DataFrame | None = None

    def load(self) -> None:
        """Load encoder, embeddings, and metadata."""
        raise NotImplementedError(
            "Implement model, embedding, and metadata loading for production use."
        )

    def search(self, request: RetrievalRequest) -> pd.DataFrame:
        """Return top-k recipe matches for a query."""
        if self.encoder is None or self.embeddings is None or self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        query_vec = self.encoder.encode([request.query])[0]
        scores = cosine_similarity(query=query_vec, matrix=self.embeddings)
        top_idx = np.argsort(-scores)[: request.top_k]
        results = self.metadata.iloc[top_idx].copy()
        results["similarity_score"] = scores[top_idx]
        results = apply_basic_filters(
            results,
            dietary_filter=request.dietary_filter,
            max_time_minutes=request.max_time_minutes,
        )
        return results.reset_index(drop=True)
