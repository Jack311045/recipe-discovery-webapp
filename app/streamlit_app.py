"""Main Streamlit entrypoint for the recipe discovery app."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.components.theme import apply_restaurant_menu_theme
from recipe_discovery.settings import get_app_title


def main() -> None:
    st.set_page_config(page_title=get_app_title(), layout="wide")
    apply_restaurant_menu_theme()

    st.markdown("<p class='menu-script-accent'>Chef's Selection</p>", unsafe_allow_html=True)
    st.title(get_app_title())
    st.markdown(
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
