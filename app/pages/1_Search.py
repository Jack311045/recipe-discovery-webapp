"""Semantic search page — wired to RetrievalService."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
from PIL import Image

from app.components.search_ui import (
    DISPLAY_MODES,
    SORT_OPTIONS,
    build_result_summary,
    get_active_tags,
    infer_tag_columns,
    sort_results_for_display,
)
from app.components.recipe_cards import render_recipe_card
from app.service_loader import get_retrieval_service
from recipe_discovery.retrieval.service import RetrievalRequest


def _format_avg(value: float | int | None, *, unit: str = "", decimals: int = 1) -> str:
    """Format optional average metrics for display cards."""
    if value is None:
        return "N/A"
    if decimals == 0:
        base = f"{float(value):.0f}"
    else:
        base = f"{float(value):.{decimals}f}"
    return f"{base} {unit}".strip()


st.set_page_config(page_title="Search Recipes", layout="wide")

if "search_results_df" not in st.session_state:
    st.session_state["search_results_df"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "last_search_mode" not in st.session_state:
    st.session_state["last_search_mode"] = ""

st.markdown(
    """
    <style>
    .search-hero {
        padding: 1rem 1.1rem;
        border-radius: 0.8rem;
        background: linear-gradient(120deg, #f6fbff 0%, #eef7f2 100%);
        border: 1px solid #d9e8dc;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="search-hero">
      <h2 style="margin:0;">🔍 Search Recipes</h2>
      <p style="margin:0.35rem 0 0 0;">Search by text or upload a dish photo, then explore polished recipe cards and frontend-only display controls.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Semantic retrieval powered by sentence-transformer embeddings + cosine similarity.")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    st.caption("These controls narrow results after semantic matching.")
    diet_options = {
        "Any": None,
        "Vegetarian": "vegetarian",
        "Vegan": "vegan",
        "Gluten-free": "gluten-free",
    }
    diet_label = st.selectbox(
        "Dietary preference",
        list(diet_options.keys()),
        help="Keeps retrieval behavior unchanged; only filters returned recipes.",
    )
    max_time = st.slider("Max cooking time (minutes)", min_value=5, max_value=180, value=60, step=5)
    max_ingredients = st.slider("Max ingredients", min_value=2, max_value=30, value=15, step=1)
    top_k = st.slider("Number of results", min_value=3, max_value=20, value=8)
    alpha = st.slider(
        "Image vs text weight (combined search)",
        min_value=0.1,
        max_value=0.9,
        value=0.75,
        step=0.05,
        help="Higher keeps the uploaded dish as the anchor. Lower lets the text steer more.",
    )

col_text, col_upload = st.columns([2, 1])

with col_text:
    query = st.text_input(
        "Describe what you want to eat",
        placeholder="quick spicy tofu dinner…",
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Or upload a dish photo",
        type=["png", "jpg", "jpeg"],
        help="Image search uses SigLIP embeddings; text search stays on SBERT.",
    )

search_clicked = st.button("Search", type="primary", use_container_width=True)

if search_clicked:
    svc = get_retrieval_service()
    request = RetrievalRequest(
        query=query,
        top_k=top_k,
        dietary_filter=diet_options[diet_label],
        max_time_minutes=max_time,
        max_ingredients=max_ingredients,
    )

    if uploaded_file is not None and query.strip():
        image = Image.open(uploaded_file)
        st.image(image, caption="Searching with image + text…", width=220)
        with st.spinner("Searching with image + text…"):
            results = svc.search_combined(query, image, request, alpha=alpha)
        search_mode = "image+text"
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Searching by this image…", width=220)
        with st.spinner("Searching with image…"):
            results = svc.search_by_image(image, request)
        search_mode = "image"
    elif query.strip():
        with st.spinner("🔎 Searching and preparing rich recipe cards…"):
            results = svc.search(request)
        search_mode = "text"
    else:
        st.warning("Please enter a search query or upload an image.")
        results = None
        search_mode = ""

    if isinstance(results, pd.DataFrame):
        st.session_state["search_results_df"] = results.copy()
        st.session_state["last_query"] = query.strip()
        st.session_state["last_search_mode"] = search_mode

results_df = st.session_state.get("search_results_df")
search_query = st.session_state.get("last_query", "")
search_mode = st.session_state.get("last_search_mode", "")

if isinstance(results_df, pd.DataFrame):
    if results_df.empty:
        st.info("No recipes matched your filters. Try relaxing the constraints.")
    else:
        if search_mode == "image+text":
            st.success(f"Found **{len(results_df)}** recipes for your image + text query: *{search_query}*")
        elif search_mode == "image":
            st.success(f"Found **{len(results_df)}** recipes for your image.")
        else:
            st.success(f"Found **{len(results_df)}** recipes for: *{search_query}*")

        summary = build_result_summary(results_df)
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Results", f"{int(summary['count'])}")
        with summary_cols[1]:
            st.metric("Avg cook time", _format_avg(summary["avg_minutes"], unit="min", decimals=0))
        with summary_cols[2]:
            st.metric("Avg rating", _format_avg(summary["avg_rating"], decimals=2))
        with summary_cols[3]:
            st.metric("Avg calories", _format_avg(summary["avg_calories"], unit="kcal", decimals=0))

        st.markdown("### Display controls")
        control_cols = st.columns([2, 2, 3])
        with control_cols[0]:
            sort_mode = st.selectbox(
                "Sort displayed results",
                SORT_OPTIONS,
                index=0,
                help="Display-only ordering. Retrieval and backend ranking are unchanged.",
            )
        with control_cols[1]:
            display_mode = st.radio(
                "Card view",
                DISPLAY_MODES,
                horizontal=True,
                index=0,
                help="Detailed shows tabs; compact keeps cards denser.",
            )
        with control_cols[2]:
            max_tags = st.slider(
                "Maximum tag chips shown per recipe",
                min_value=3,
                max_value=12,
                value=8,
                help="Tag chips are visual only and do not affect ranking/filter behavior.",
            )

        display_df = sort_results_for_display(results_df, sort_mode)
        tag_columns = infer_tag_columns(display_df)

        has_proj = "x_proj" in display_df.columns
        if has_proj:
            st.caption(
                "📍 PCA coordinates are attached — check the Embedding Map page "
                "to see where these land!"
            )

        for rank, (_, row) in enumerate(display_df.iterrows(), start=1):
            row_dict = row.to_dict()
            row_dict["_active_tags"] = get_active_tags(
                row_dict,
                tag_columns,
                max_tags=max_tags,
            )
            render_recipe_card(
                row_dict,
                rank=rank,
                display_mode=display_mode,
            )
