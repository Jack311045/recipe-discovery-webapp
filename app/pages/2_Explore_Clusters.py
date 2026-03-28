"""Cluster exploration page."""

from __future__ import annotations

import streamlit as st

st.title("Explore Clusters")
st.caption("Browse recipe groups discovered from embedding space.")

k = st.slider("Number of clusters", min_value=2, max_value=20, value=8)

st.info(
    "This page will eventually show cluster summaries such as quick meals, desserts, "
    "high-protein recipes, or cuisine-specific groups."
)
st.write({"selected_cluster_count": k})
