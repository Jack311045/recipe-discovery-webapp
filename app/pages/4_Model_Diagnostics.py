"""Model diagnostics page."""

from __future__ import annotations

import streamlit as st

st.title("Model Diagnostics")
st.caption("Evaluation summaries for retrieval, clustering, regression, and projection.")

st.markdown(
    """
    Placeholder diagnostics to include later:
    - retrieval relevance metrics
    - cluster size and cohesion summaries
    - regression MAE / RMSE / R²
    - reconstruction loss for the autoencoder
    """
)
