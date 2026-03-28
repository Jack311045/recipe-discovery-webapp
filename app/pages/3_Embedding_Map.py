"""2D embedding map page."""

from __future__ import annotations

import streamlit as st

st.title("Embedding Map")
st.caption("2D projection of recipe embeddings.")

projection = st.radio("Projection method", ["PCA", "Autoencoder"], horizontal=True)

st.warning(
    "Use axis labels like PC1/PC2 for PCA or Latent 1/Latent 2 for autoencoder. "
    "Do not label them as time or nutrition unless you explicitly learn supervised axes."
)
st.write({"projection_method": projection})
