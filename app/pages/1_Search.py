"""Semantic search page — wired to RetrievalService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import streamlit as st

from app.components.search_ui import (
    DISPLAY_MODES,
    SORT_OPTIONS,
    build_result_summary,
    get_active_tags,
    infer_tag_columns,
    sort_results_for_display,
)
from app.service_loader import get_retrieval_service
from app.components.recipe_cards import render_recipe_card
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
      <p style="margin:0.35rem 0 0 0;">Discover recipes by intent with semantic search, then explore richer card details without changing retrieval logic.</p>
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
    use_time_limit = st.checkbox("Apply max cooking time filter", value=False)
    max_time: int | None = None
    if use_time_limit:
        max_time = st.slider(
            "Max cooking time (minutes)",
            min_value=5,
            max_value=180,
            value=60,
            step=5,
        )

    use_ingredient_limit = st.checkbox("Apply max ingredients filter", value=False)
    max_ingredients: int | None = None
    if use_ingredient_limit:
        max_ingredients = st.slider(
            "Max ingredients",
            min_value=2,
            max_value=30,
            value=15,
            step=1,
        )

    st.caption("Time and ingredient filters are optional and disabled by default.")
    top_k = st.slider("Number of results", min_value=3, max_value=20, value=10)

query_col, button_col = st.columns([5, 1])
with query_col:
    query = st.text_input(
        "Describe what you want to eat",
        placeholder="quick spicy tofu dinner…",
        key="search_query_input",
    )
with button_col:
    st.write(" ")
    search_clicked = st.button("Search", type="primary", use_container_width=True)

if search_clicked:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("🔎 Searching and preparing rich recipe cards…"):
            svc = get_retrieval_service()
            request = RetrievalRequest(
                query=query,
                top_k=top_k,
                dietary_filter=diet_options[diet_label],
                max_time_minutes=max_time,
                max_ingredients=max_ingredients,
            )
            results = svc.search(request)

        st.session_state["search_results_df"] = results.copy()
        st.session_state["last_query"] = query.strip()

results_df = st.session_state.get("search_results_df")
search_query = st.session_state.get("last_query", query.strip())

if isinstance(results_df, pd.DataFrame):
    if results_df.empty:
        st.info("No recipes matched your filters. Try relaxing the constraints.")
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
                "📍 PCA coordinates are attached — check the Embedding Map page to see where these land!"
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
