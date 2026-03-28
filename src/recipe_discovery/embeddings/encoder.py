"""Transformer-based recipe text encoder."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for dense text embedding."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32


class RecipeEncoder:
    """Wrapper around a sentence-transformer style model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.model = None

    def load(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.model = SentenceTransformer(self.config.model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode recipe texts into dense vectors."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)
