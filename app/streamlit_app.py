"""Main Streamlit entrypoint and page router for the recipe discovery app."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.components.theme import apply_restaurant_menu_theme
from app.service_loader import start_retrieval_service_warmup


def main() -> None:
    st.set_page_config(page_title="Recipe Discovery", layout="wide")
    apply_restaurant_menu_theme()
    start_retrieval_service_warmup()

    pages = [
        st.Page(APP_DIR / "pages" / "1_Search.py", title="Search", default=True),
        st.Page(APP_DIR / "pages" / "5_Shopping_List.py", title="Shopping List"),
        st.Page(APP_DIR / "pages" / "2_Explore_Clusters.py", title="Explore Clusters"),
        st.Page(APP_DIR / "pages" / "3_Embedding_Map.py", title="Embedding Map"),
        st.Page(APP_DIR / "pages" / "4_Model_Diagnostics.py", title="Model Diagnostics"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
