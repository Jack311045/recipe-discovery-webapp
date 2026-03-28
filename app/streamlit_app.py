"""Main Streamlit entrypoint for the recipe discovery app."""

from __future__ import annotations

import streamlit as st

from recipe_discovery.settings import get_app_title


def main() -> None:
    st.set_page_config(page_title=get_app_title(), layout="wide")
    st.title(get_app_title())
    st.write(
        "Semantic recipe retrieval, clustering, ranking, and 2D visualization."
    )

    with st.container(border=True):
        st.subheader("Checkpoint status")
        st.write(
            "This app is currently a professional scaffold. The UI pages, module "
            "boundaries, and training scripts are in place so the repo is runnable "
            "and ready for implementation."
        )

    st.markdown(
        """
        ### Included modules
        - Embedding generation
        - Nearest-neighbor retrieval with cosine similarity
        - K-means clustering
        - Regression ranking
        - PCA and autoencoder projection
        - Evaluation utilities
        """
    )


if __name__ == "__main__":
    main()
