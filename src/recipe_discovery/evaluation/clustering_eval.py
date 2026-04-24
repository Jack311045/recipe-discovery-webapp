"""Clustering diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Return counts per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts, strict=False)}


def silhouette_summary(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int | None = 5000,
    random_state: int = 42,
    metric: str = "euclidean",
) -> dict[str, float | int | None]:
    """Compute silhouette score with optional subsampling.

    Silhouette is O(n^2) in memory, so on a 180K-row Food.com embedding matrix it
    would require a 240 GB pairwise-distance buffer. We subsample uniformly when
    ``sample_size`` is smaller than the input. Returns the score together with
    the actual sample size used, for transparency in the metadata artifact.

    Returns
    -------
    dict with keys ``silhouette_score`` (float), ``sample_size`` (int), and
    ``metric`` (str). ``silhouette_score`` is ``None`` when fewer than two
    distinct clusters are present (silhouette is undefined there).
    """
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)

    if len(np.unique(labels)) < 2:
        return {"silhouette_score": None, "sample_size": int(len(labels)), "metric": metric}

    n = len(labels)
    if sample_size is not None and sample_size < n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=int(sample_size), replace=False)
        sub_embeddings = embeddings[idx]
        sub_labels = labels[idx]
        # Subsample may drop a rare cluster entirely; re-check.
        if len(np.unique(sub_labels)) < 2:
            return {"silhouette_score": None, "sample_size": int(sample_size), "metric": metric}
        score = float(silhouette_score(sub_embeddings, sub_labels, metric=metric))
        return {"silhouette_score": score, "sample_size": int(sample_size), "metric": metric}

    score = float(silhouette_score(embeddings, labels, metric=metric))
    return {"silhouette_score": score, "sample_size": int(n), "metric": metric}
