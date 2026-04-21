"""Abstract base class for dimensionality reduction modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseReducer(ABC):
    """Abstract base class for 2D projection reducers."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the reducer to the embeddings."""

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to 2D.
        
        Returns
        -------
        np.ndarray
            Array of shape (N, 2).
        """

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform the embeddings."""
        self.fit(embeddings)
        return self.transform(embeddings)
