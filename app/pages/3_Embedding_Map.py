"""2D Embedding Map — shows PCA/autoencoder projections from the dimensionality
reduction module, optionally colored by k-means cluster.

Cluster coloring overlays semantic groupings on top of the geometric 2D
projection. The two views answer different questions:

- Projection (geometry): "Where do recipes sit in semantic space?"
- Cluster colors (semantics): "Which recipes the algorithm decided are similar?"

Combined, they let users see whether the clustering decisions agree with the
geometric structure of the embeddings. This fulfils the proposal's goal of
showing 'how clusters relate to one another' in semantic space.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.plots import scatter_2d_with_highlights
from app.service_loader import get_retrieval_service
from recipe_discovery.clustering.kmeans import KMeans
from recipe_discovery.clustering.labels import name_clusters
from recipe_discovery.retrieval.service import RetrievalRequest
from recipe_discovery.settings import ARTIFACTS_DIR

st.set_page_config(page_title="Embedding Map", layout="wide")
st.title("🗺️ Embedding Map")
st.caption(
    "Each point is a recipe projected into 2D using your dimensionality reduction module. "
    "Toggle cluster coloring to see how semantic groups distribute across the projection. "
    "Run a search to highlight where matching recipes land."
)

projection_method = "PCA"


# ---------------------------------------------------------------------------
# Cluster overlay (loaded lazily, cached across reruns)
# ---------------------------------------------------------------------------

# A small, stable concept vocabulary for human-readable cluster names. Mirrors
# the vocabulary used in the Explore Clusters page so naming stays consistent.
_CONCEPT_VOCABULARY: tuple[str, ...] = (
    "dessert", "breakfast", "dinner", "appetizer", "snack",
    "soup", "salad", "pizza", "pasta", "burger", "sandwich",
    "taco", "sushi", "curry", "cake", "cookies", "bread", "pie",
    "chicken", "beef", "pork", "seafood", "vegetables", "fruit",
    "italian", "chinese", "japanese", "mexican", "indian", "thai",
    "french", "korean", "grilled", "fried", "roasted",
)


@st.cache_resource(show_spinner="Loading cluster assignments…")
def _load_cluster_assignments() -> pd.DataFrame | None:
    """Return a DataFrame with columns ``recipe_id``, ``cluster``, ``cluster_name``
    or ``None`` if cluster artifacts are unavailable.

    The resulting frame is keyed on ``recipe_id`` for left-merging onto the
    projection DataFrame.
    """
    kmeans_path = ARTIFACTS_DIR / "kmeans.joblib"
    embeddings_path = ARTIFACTS_DIR / "recipe_embeddings.npy"
    ids_path = ARTIFACTS_DIR / "recipe_ids.csv"
    if not (kmeans_path.exists() and embeddings_path.exists() and ids_path.exists()):
        return None

    try:
        model = KMeans.load(kmeans_path)
        embeddings = np.load(embeddings_path)
        ids = pd.read_csv(ids_path)["recipe_id"].astype(int).values
    except (OSError, ValueError, KeyError):
        return None

    if embeddings.shape[0] != len(ids) or embeddings.shape[1] != model.centroids_.shape[1]:
        # Mismatched artifacts — better to silently fall back than to colour
        # points with the wrong cluster.
        return None

    cluster_labels = model.predict(embeddings)
    return pd.DataFrame(
        {"recipe_id": ids, "cluster": cluster_labels.astype(int)}
    )


@st.cache_data(show_spinner="Naming clusters…")
def _build_cluster_name_map(
    projection_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    text_columns: tuple[str, ...],
) -> dict[int, str]:
    """Generate a cluster_id -> human-readable name mapping using the
    distinctive-word naming pipeline. Returns a dict like
    ``{0: "pasta: spaghetti, lasagna", 1: "cake: pie, muffins", ...}``.
    """
    available_text = [c for c in text_columns if c in projection_df.columns]
    if not available_text:
        # Projection frame has no text fields — fall back to "Cluster N" labels.
        return {int(c): f"Cluster {int(c)}" for c in cluster_df["cluster"].unique()}

    merged = projection_df.merge(cluster_df, on="recipe_id", how="inner")
    if merged.empty:
        return {int(c): f"Cluster {int(c)}" for c in cluster_df["cluster"].unique()}

    named = name_clusters(
        merged,
        text_columns=tuple(available_text),
        vocabulary=list(_CONCEPT_VOCABULARY),
        top_n=6,
        min_doc_freq=5,
    )
    return {int(cid): str(info["name"]) for cid, info in named.items()}


# ---------------------------------------------------------------------------
# Load projections
# ---------------------------------------------------------------------------

svc = get_retrieval_service()
all_proj = svc.get_all_projections()

if all_proj.empty:
    st.error(
        "No 2D projections found. Run `scripts/fit_pca.py` (or `scripts/train_autoencoder.py`) "
        "to generate `pca_projection.npy` / `projections_2d.npy` in `data/artifacts/`."
    )
    st.stop()

if projection_method == "PCA":
    x_label, y_label = "PC1", "PC2"
else:
    x_label, y_label = "Latent Dim 1", "Latent Dim 2"
    ae_path = ARTIFACTS_DIR / "projections_2d.npy"
    if not ae_path.exists():
        st.warning(
            "Autoencoder projections (`projections_2d.npy`) not found — "
            "showing PCA projections as fallback. "
            "Run `scripts/train_autoencoder.py` to generate them."
        )
        x_label, y_label = "PC1 (fallback)", "PC2 (fallback)"


# ---------------------------------------------------------------------------
# Attach cluster labels (when artifacts are present)
# ---------------------------------------------------------------------------

cluster_df = _load_cluster_assignments()
cluster_name_map: dict[int, str] = {}

if cluster_df is not None and "recipe_id" in all_proj.columns:
    all_proj = all_proj.merge(cluster_df, on="recipe_id", how="left")
    cluster_name_map = _build_cluster_name_map(
        all_proj, cluster_df, text_columns=("name", "description")
    )
    if cluster_name_map:
        all_proj["cluster_name"] = all_proj["cluster"].map(
            lambda c: cluster_name_map.get(int(c), f"Cluster {int(c)}") if pd.notna(c) else None
        )


# ---------------------------------------------------------------------------
# Color toggle (only shown when cluster column is available)
# ---------------------------------------------------------------------------

color_by_cluster = False
if "cluster" in all_proj.columns:
    color_mode = st.radio(
        "Color markers by", ["Default (grey)", "Cluster labels"], horizontal=True
    )
    color_by_cluster = color_mode == "Cluster labels"
    if color_by_cluster and cluster_name_map:
        # Show the cluster -> name legend so the colors are interpretable.
        with st.expander("🎨 Cluster legend", expanded=False):
            for cid in sorted(cluster_name_map.keys()):
                st.markdown(f"- **Cluster {cid}**: {cluster_name_map[cid]}")
elif cluster_df is None:
    st.caption(
        "💡 To enable cluster coloring, run `python scripts/train_kmeans.py` to generate "
        "`data/artifacts/kmeans.joblib`."
    )


# ---------------------------------------------------------------------------
# Performance controls
# ---------------------------------------------------------------------------

max_points = st.slider(
    "Background points",
    min_value=1000,
    max_value=min(100000, max(len(all_proj), 1000)),
    value=min(20000, len(all_proj)),
    step=1000,
    help="Reduce points to speed up rendering.",
)
show_background_hover = st.checkbox(
    "Show hover labels for background points",
    value=True,
    help="Disabling hover improves rendering speed for large point clouds.",
)

if len(all_proj) > max_points:
    # Sampling keeps interactions snappy while preserving the overall shape.
    all_proj = all_proj.sample(n=max_points, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Search highlight
# ---------------------------------------------------------------------------

with st.expander("🔍 Highlight search results on the map", expanded=False):
    highlight_query = st.text_input(
        "Enter a query to highlight matching recipes (overrides Search page)",
        key="map_search",
    )
    top_k_map = st.slider("Number of results to highlight", 1, 20, 8, key="map_topk")
    do_highlight = st.button("Highlight", key="map_btn")

highlight_df = pd.DataFrame()

if do_highlight and highlight_query.strip():
    with st.spinner("Searching…"):
        results = svc.search(RetrievalRequest(query=highlight_query, top_k=top_k_map))
    if not results.empty and "x_proj" in results.columns:
        highlight_df = results.dropna(subset=["x_proj", "y_proj"])
        st.success(f"Highlighting {len(highlight_df)} results for: *{highlight_query}*")
    else:
        st.info("No results with projection data found.")
elif "search_results_df" in st.session_state and isinstance(
    st.session_state["search_results_df"], pd.DataFrame
):
    base_highlight_df = st.session_state["search_results_df"]
    if not base_highlight_df.empty and "x_proj" in base_highlight_df.columns:
        highlight_df = base_highlight_df.dropna(subset=["x_proj", "y_proj"])
        last_query = st.session_state.get("last_query", "")
        if last_query:
            st.info(f"Showing highlighted results from your Search page query: *{last_query}*")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

# Prefer the human-readable cluster name as the color column when available;
# fall back to the integer cluster id otherwise.
color_column: str | None = None
if color_by_cluster:
    color_column = "cluster_name" if "cluster_name" in all_proj.columns else "cluster"

fig = scatter_2d_with_highlights(
    background_df=all_proj,
    highlight_df=highlight_df,
    x="x_proj",
    y="y_proj",
    hover_name="name",
    x_label=x_label,
    y_label=y_label,
    title=f"Recipe Embedding Map ({projection_method})",
    background_hover=show_background_hover,
    color=color_column,
)
st.plotly_chart(fig, use_container_width=True)


with st.expander("ℹ️ What am I looking at?"):
    st.markdown(
        f"""
        - Each **point** is one of the {len(all_proj):,} loaded recipes, projected from a
          384-dimensional sentence-transformer embedding space down to 2D
          using **{projection_method}**.
        - **Orange stars** are your search results — you can see how semantically similar
          recipes cluster together in this 2D space.
        - When **Cluster labels** coloring is on, each color is one of the K-means clusters
          (named automatically via TF-IDF over recipe text). Points of the same color are
          recipes our clustering algorithm decided are semantically similar.
        - For PCA, the axes (PC1, PC2) capture the two directions of maximum variance in
          the embedding space. They don't directly correspond to any human-interpretable
          feature like cuisine or cook time.
        - PCA explained variance: only ~14.7% — meaning the 2D picture captures a portion
          of the full structure. The autoencoder can learn a nonlinear manifold that may
          capture more.
        """
    )
