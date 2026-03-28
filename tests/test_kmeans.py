"""Tests for k-means."""

from __future__ import annotations

import numpy as np

from recipe_discovery.clustering.kmeans import KMeans


def test_kmeans_fit_predict_returns_labels() -> None:
    x = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    model = KMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(x)
    assert len(labels) == len(x)
