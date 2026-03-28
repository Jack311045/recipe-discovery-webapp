"""K-means clustering on recipe embeddings."""

from __future__ import annotations

import numpy as np


class KMeans:
    """A small educational implementation of k-means."""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "KMeans":
        """Fit centroids to embedding matrix x."""
        rng = np.random.default_rng(self.random_state)
        initial_idx = rng.choice(len(x), size=self.n_clusters, replace=False)
        centroids = x[initial_idx].copy()

        for _ in range(self.max_iter):
            distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)

            new_centroids = np.vstack(
                [
                    x[labels == cluster_id].mean(axis=0)
                    if np.any(labels == cluster_id)
                    else centroids[cluster_id]
                    for cluster_id in range(self.n_clusters)
                ]
            )

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        self.centroids = centroids
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        if self.centroids is None:
            raise RuntimeError("Model is not fitted.")

        distances = ((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster assignments."""
        self.fit(x)
        return self.predict(x)
