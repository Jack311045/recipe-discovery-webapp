"""Transformer-based recipe text encoder."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for dense text embedding."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True


class RecipeEncoder:
    """Wrapper around a sentence-transformer model.

    Usage::

        enc = RecipeEncoder(EmbeddingConfig())
        enc.load()
        vecs = enc.encode(["some recipe text", ...])
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.model = None

    def load(self) -> None:
        """Load the embedding model into memory."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run `pip install -r requirements.txt`."
            ) from exc

        logger.info("Loading model: %s", self.config.model_name)
        self.model = SentenceTransformer(self.config.model_name)
        logger.info("Model loaded. Embedding dim = %d", self.model.get_embedding_dimension())

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the loaded model's output."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call `load()` first.")
        return self.model.get_embedding_dimension()

    def encode(self, texts: Iterable[str], *, show_progress: bool = True) -> np.ndarray:
        """Encode recipe texts into dense vectors.

        Parameters
        ----------
        texts:
            Iterable of recipe text strings.
        show_progress:
            Whether to display a progress bar during encoding.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_texts, embedding_dim)``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        text_list = list(texts)
        logger.info("Encoding %d texts (batch_size=%d).", len(text_list), self.config.batch_size)

        embeddings = self.model.encode(
            text_list,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
        )
        result = np.asarray(embeddings)
        logger.info("Encoding complete. Shape: %s", result.shape)
        return result
