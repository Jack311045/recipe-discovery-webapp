"""2D Embedding Map — shows PCA projections from the dimensionality reduction module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st

from app.service_loader import get_retrieval_service
from app.components.plots import scatter_2d_with_highlights, scatter_2d
from recipe_discovery.retrieval.service import RetrievalRequest

st.set_page_config(page_title="Embedding Map", layout="wide")
st.title("🗺️ Embedding Map")
st.caption(
    "Each point is a recipe projected into 2D using your dimensionality reduction module. "
    "Run a search to highlight where matching recipes land."
)

projection_method = st.radio("Projection method", ["PCA", "Autoencoder"], horizontal=True)

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
    from recipe_discovery.settings import ARTIFACTS_DIR
    ae_path = ARTIFACTS_DIR / "projections_2d.npy"
    if not ae_path.exists():
        st.warning(
            "Autoencoder projections (`projections_2d.npy`) not found — "
            "showing PCA projections as fallback. Run `scripts/train_autoencoder.py` to generate them."
        )
        x_label, y_label = "PC1 (fallback)", "PC2 (fallback)"

# Optional: search query to highlight results
with st.expander("🔍 Highlight search results on the map", expanded=False):
    highlight_query = st.text_input("Enter a query to highlight matching recipes", key="map_search")
    top_k_map = st.slider("Number of results to highlight", 1, 20, 8, key="map_topk")
    do_highlight = st.button("Highlight", key="map_btn")

highlight_df = __import__("pandas").DataFrame()

if do_highlight and highlight_query.strip():
    with st.spinner("Searching…"):
        results = svc.search(RetrievalRequest(query=highlight_query, top_k=top_k_map))
    if not results.empty and "x_proj" in results.columns:
        highlight_df = results[["name", "x_proj", "y_proj"]].dropna()
        st.success(f"Highlighting {len(highlight_df)} results for: *{highlight_query}*")
    else:
        st.info("No results with projection data found.")

fig = scatter_2d_with_highlights(
    background_df=all_proj,
    highlight_df=highlight_df,
    x="x_proj",
    y="y_proj",
    hover_name="name",
    x_label=x_label,
    y_label=y_label,
    title=f"Recipe Embedding Map ({projection_method})",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("ℹ️ What am I looking at?"):
    st.markdown(
        f"""
        - Each **grey dot** is one of the {len(all_proj):,} loaded recipes, projected from a
          384-dimensional sentence-transformer embedding space down to 2D using **{projection_method}**.
        - **Orange stars** are your search results — you can see how semantically similar recipes
          cluster together in this 2D space.
        - For PCA, the axes (PC1, PC2) capture the two directions of maximum variance in the embedding space.
          They don't directly correspond to any human-interpretable feature like cuisine or cook time.
        - PCA explained variance: only ~14.7% — meaning the 2D picture captures a portion of the
          full structure. The autoencoder can learn a nonlinear manifold that may capture more.
        """
    )
