"""Cluster filtering helper for the Search page.

Loads the saved k-means artifact and exposes:

- ``load_cluster_options()``: returns ``[(label, cluster_id), ...]`` for a
  Streamlit selectbox, with auto-generated names.
- ``filter_by_cluster(results_df, cluster_id)``: post-search filter step that
  keeps only rows whose recipe_id falls in the chosen cluster.

Designed to be a no-op when the cluster artifact is missing — the page
gracefully falls back to "All clusters" mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recipe_discovery.clustering.kmeans import KMeans
from recipe_discovery.clustering.labels import name_clusters
from recipe_discovery.settings import ARTIFACTS_DIR, DATA_PROCESSED_DIR


_CONCEPT_VOCABULARY: tuple[str, ...] = (
    "dessert", "breakfast", "dinner", "appetizer", "snack",
    "soup", "salad", "pizza", "pasta", "burger", "sandwich",
    "taco", "sushi", "curry", "cake", "cookies", "bread", "pie",
    "chicken", "beef", "pork", "seafood", "vegetables", "fruit",
    "italian", "chinese", "japanese", "mexican", "indian", "thai",
    "french", "korean", "grilled", "fried", "roasted",
)


def _processed_csv_path() -> Path | None:
    """Find the processed-recipes CSV regardless of filename casing variations."""
    candidates = [
        DATA_PROCESSED_DIR / "Processed_data_updated2.csv",
        DATA_PROCESSED_DIR / "processed_data_updated2.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def cluster_artifacts_available() -> bool:
    """Quick check used by the page to decide whether to render the dropdown."""
    return (
        (ARTIFACTS_DIR / "kmeans.joblib").exists()
        and (ARTIFACTS_DIR / "recipe_embeddings.npy").exists()
        and (ARTIFACTS_DIR / "recipe_ids.csv").exists()
    )


def load_cluster_assignments() -> pd.DataFrame | None:
    """Return ``DataFrame[recipe_id (str), cluster (int)]`` or ``None`` on failure."""
    if not cluster_artifacts_available():
        return None
    try:
        model = KMeans.load(ARTIFACTS_DIR / "kmeans.joblib")
        embeddings = np.load(ARTIFACTS_DIR / "recipe_embeddings.npy")
        ids_series = pd.read_csv(ARTIFACTS_DIR / "recipe_ids.csv")["recipe_id"]
    except (OSError, ValueError, KeyError):
        return None

    if embeddings.shape[0] != len(ids_series) or embeddings.shape[1] != model.centroids_.shape[1]:
        return None

    cluster_labels = model.predict(embeddings)
    return pd.DataFrame(
        {
            "recipe_id": ids_series.astype(str).values,
            "cluster": cluster_labels.astype(int),
        }
    )


def load_cluster_name_map(cluster_df: pd.DataFrame) -> dict[int, str]:
    """Generate cluster_id -> human-readable name using TF-IDF over recipe text.

    Uses the same naming pipeline as the Explore Clusters and Embedding Map
    pages so labels stay consistent everywhere.
    """
    csv_path = _processed_csv_path()
    if csv_path is None:
        return {int(c): f"Cluster {int(c)}" for c in cluster_df["cluster"].unique()}

    try:
        recipes = pd.read_csv(
            csv_path,
            usecols=["recipe_id", "name", "description"],
            low_memory=False,
        )
    except (FileNotFoundError, ValueError):
        return {int(c): f"Cluster {int(c)}" for c in cluster_df["cluster"].unique()}

    recipes["recipe_id"] = recipes["recipe_id"].astype(str)
    merged = recipes.merge(cluster_df, on="recipe_id", how="inner")
    if merged.empty:
        return {int(c): f"Cluster {int(c)}" for c in cluster_df["cluster"].unique()}

    named = name_clusters(
        merged,
        text_columns=("name", "description"),
        vocabulary=list(_CONCEPT_VOCABULARY),
        top_n=6,
        min_doc_freq=5,
    )
    return {int(cid): str(info["name"]) for cid, info in named.items()}


def build_dropdown_options(
    cluster_df: pd.DataFrame, name_map: dict[int, str]
) -> list[tuple[str, int | None]]:
    """Return list of ``(label, cluster_id_or_None)`` for the selectbox.

    ``None`` represents the "All clusters" sentinel; numeric cluster ids select
    a single cluster. Sorted by cluster size (largest first) so popular
    categories appear first in the dropdown.
    """
    sizes = cluster_df["cluster"].value_counts()
    options: list[tuple[str, int | None]] = [("All clusters", None)]
    for cluster_id in sizes.index:
        cid = int(cluster_id)
        label = name_map.get(cid, f"Cluster {cid}")
        size = int(sizes.loc[cluster_id])
        options.append((f"{label} ({size:,})", cid))
    return options


def filter_results_by_cluster(
    results: pd.DataFrame,
    cluster_id: int,
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only the rows in ``results`` whose ``recipe_id`` is in ``cluster_id``."""
    if "recipe_id" not in results.columns:
        return results
    target_ids = set(
        cluster_df.loc[cluster_df["cluster"] == cluster_id, "recipe_id"].astype(str)
    )
    if not target_ids:
        return results.iloc[0:0]
    mask = results["recipe_id"].astype(str).isin(target_ids)
    return results.loc[mask].reset_index(drop=True)


def cluster_filter_top_k_multiplier(
    cluster_id: int, cluster_df: pd.DataFrame, *, max_multiplier: int = 10
) -> int:
    """Heuristic: when filtering to a small cluster after retrieval, we need to
    over-fetch from the search step so the post-filter result set is non-empty.

    Returns a multiplier ``m``: caller should set ``top_k = base_top_k * m`` for
    the underlying retrieval call. Capped to avoid pathological retrieval sizes.
    """
    if cluster_id is None:
        return 1
    cluster_size = int((cluster_df["cluster"] == cluster_id).sum())
    if cluster_size == 0:
        return 1
    total = int(len(cluster_df))
    fraction = cluster_size / total
    if fraction <= 0:
        return max_multiplier
    multiplier = int(round(1.0 / fraction))
    return max(1, min(multiplier, max_multiplier))
