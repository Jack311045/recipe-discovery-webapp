"""Semantic search page — wired to RetrievalService."""

from __future__ import annotations

import random
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
from app.components.recipe_cards import parse_ingredients, render_recipe_card
from app.components.shopping_list import (
    add_ingredients_to_shopping_list,
    ensure_shopping_list_state,
    get_shopping_list_count,
)
from app.components.theme import apply_restaurant_menu_theme
from app.service_loader import get_retrieval_service
from recipe_discovery.retrieval.service import RetrievalRequest

LANDING_QUERIES = [
    "quick weeknight dinner",
    "healthy breakfast ideas",
    "easy comfort food",
    "simple vegetarian recipes",
]


def _format_avg(value: float | int | None, *, unit: str = "", decimals: int = 1) -> str:
    """Format optional average metrics for display cards."""
    if value is None:
        return "N/A"
    if decimals == 0:
        base = f"{float(value):.0f}"
    else:
        base = f"{float(value):.{decimals}f}"
    return f"{base} {unit}".strip()


def _initialize_session_state() -> None:
    """Initialize search-page session state keys."""
    defaults = {
        "search_results_df": None,
        "last_query": "",
        "last_search_mode": "",
        "search_query_input": "",
        "search_history": [],
        "landing_results_df": None,
        "landing_query": "",
        "history_search_requested": False,
        "feedback_query_vec": None,
        "feedback_excluded_ids": set(),
        "feedback_active_request": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    ensure_shopping_list_state()


def _reset_feedback() -> None:
    """Clear relevance-feedback state for a new search context."""
    st.session_state["feedback_query_vec"] = None
    st.session_state["feedback_excluded_ids"] = set()
    st.session_state["feedback_active_request"] = None


def _add_to_history(query: str) -> None:
    """Store a unique recent search query for this browser session."""
    query = query.strip()
    if not query:
        return

    history = [
        item
        for item in st.session_state["search_history"]
        if item.lower() != query.lower()
    ]
    st.session_state["search_history"] = [query, *history][:5]


def _select_history_query(query: str) -> None:
    """Populate the text input from history and submit it on the next rerun."""
    st.session_state["search_query_input"] = query
    st.session_state["history_search_requested"] = True


def _render_search_history() -> None:
    """Render recent search chips below the text input."""
    history = st.session_state["search_history"]
    if not history:
        return

    st.caption("Recent searches")
    cols = st.columns(min(len(history), 5))
    for idx, query in enumerate(history):
        label = query if len(query) <= 24 else f"{query[:21]}..."
        with cols[idx]:
            st.button(
                label,
                key=f"history_query_{idx}",
                use_container_width=True,
                on_click=_select_history_query,
                args=(query,),
            )


def _render_skeleton_card() -> None:
    """Render a lightweight placeholder card while retrieval runs."""
    st.markdown(
        """
        <div class="result-skeleton">
          <div class="sk-img"></div>
          <div class="sk-body">
            <div class="sk-line sk-title"></div>
            <div class="sk-line sk-meta"></div>
            <div class="sk-line sk-short"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_skeleton_grid(count: int, *, label: str = "Searching recipes...") -> None:
    """Render skeleton cards into a reserved result slot."""
    st.markdown(f"**{label}**")
    for _ in range(count):
        _render_skeleton_card()


def _load_landing_results() -> pd.DataFrame:
    """Load and cache first-run landing results for this session."""
    if st.session_state["landing_results_df"] is None:
        svc = get_retrieval_service()
        query = random.choice(LANDING_QUERIES)
        request = RetrievalRequest(
            query=query,
            top_k=8,
            max_time_minutes=45,
            max_ingredients=15,
            min_rating=4.0,
        )
        results = svc.search(request)
        if results.empty:
            fallback_request = RetrievalRequest(
                query=query,
                top_k=8,
                max_time_minutes=45,
                max_ingredients=15,
            )
            results = svc.search(fallback_request)

        if "rating" in results.columns:
            sort_columns = ["rating"]
            sort_ascending = [False]
            if "similarity_score" in results.columns:
                sort_columns.append("similarity_score")
                sort_ascending.append(False)
            results = results.sort_values(
                by=sort_columns,
                ascending=sort_ascending,
                kind="mergesort",
            ).reset_index(drop=True)

        st.session_state["landing_results_df"] = results.copy()
        st.session_state["landing_query"] = query

    return st.session_state["landing_results_df"]


def _render_landing_state() -> None:
    """Render first-run recipe cards before the user searches."""
    landing_placeholder = st.empty()
    if st.session_state["landing_results_df"] is None:
        with landing_placeholder.container():
            _render_skeleton_grid(8, label="Preparing popular picks...")

    landing_df = _load_landing_results()
    landing_placeholder.empty()

    if landing_df.empty:
        return

    st.markdown("### Popular starting points")
    st.caption(f"Showing highly rated matches for: {st.session_state['landing_query']}")

    tag_columns = infer_tag_columns(landing_df)
    for rank, (_, row) in enumerate(landing_df.iterrows(), start=1):
        row_dict = row.to_dict()
        row_dict["_active_tags"] = get_active_tags(row_dict, tag_columns, max_tags=8)
        render_recipe_card(
            row_dict,
            rank=rank,
            display_mode="Compact",
            on_add_to_shopping_list=_add_recipe_to_shopping_list,
            widget_key_prefix="landing",
        )


def _add_recipe_to_shopping_list(recipe: dict[str, object]) -> None:
    """Merge recipe ingredients into shopping-list session state."""
    ingredients = parse_ingredients(recipe.get("ingredients"))
    if not ingredients:
        st.info("This recipe has no ingredient list to add.")
        return

    source_recipe = str(recipe.get("name") or recipe.get("recipe_id") or "Recipe")
    added_count, merged_count = add_ingredients_to_shopping_list(
        ingredients,
        source_recipe=source_recipe,
    )
    if added_count == 0 and merged_count == 0:
        st.info("No valid ingredient lines were found for this recipe.")
        return

    total_count = get_shopping_list_count()
    message = (
        f"Shopping list updated: +{added_count} new"
        f"{', merged ' + str(merged_count) if merged_count else ''}. "
        f"Total items: {total_count}."
    )
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        st.success(message)


def _on_negative_feedback(recipe_id: str) -> None:
    """Apply negative feedback to the active text-search results."""
    excluded = st.session_state["feedback_excluded_ids"]
    excluded.add(str(recipe_id))

    current = st.session_state.get("search_results_df")
    query_vec = st.session_state.get("feedback_query_vec")
    request = st.session_state.get("feedback_active_request")
    if query_vec is None or request is None:
        if isinstance(current, pd.DataFrame) and "recipe_id" in current.columns:
            keep = current["recipe_id"].astype(str) != str(recipe_id)
            st.session_state["search_results_df"] = current.loc[keep].reset_index(drop=True)
        return

    svc = get_retrieval_service()
    results = svc.search_with_negative_feedback(
        request,
        query_vec=query_vec,
        negative_recipe_ids=excluded,
        alpha=0.3,
    )
    st.session_state["search_results_df"] = results.copy()


st.set_page_config(page_title="Search Recipes", layout="wide")
apply_restaurant_menu_theme()

_initialize_session_state()

st.markdown(
    """
    <style>
    .result-skeleton {
        display: grid;
        grid-template-columns: 260px 1fr;
        gap: 1rem;
        padding: 1rem;
        border: 1px solid #c8b38a;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(255, 250, 241, 0.95) 0%, rgba(251, 241, 223, 0.95) 100%);
        box-shadow: 0 6px 18px rgba(85, 58, 38, 0.1);
        margin-bottom: 0.8rem;
    }
    .sk-img,
    .sk-line {
        background: #e8d9bb;
        animation: pulse 1.2s ease-in-out infinite;
    }
    .sk-img {
        height: 150px;
        border-radius: 10px;
    }
    .sk-line {
        height: 14px;
        border-radius: 999px;
        margin-bottom: 0.75rem;
    }
    .sk-title { width: 70%; height: 22px; }
    .sk-meta { width: 45%; }
    .sk-short { width: 58%; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.45; }
    }
    @media (max-width: 700px) {
        .result-skeleton {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="search-hero">
            <p class="search-kicker">Today's Menu</p>
            <h2>Search Recipes</h2>
            <p>Search by text or upload a dish photo, then explore polished recipe cards and frontend-only display controls.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Semantic retrieval powered by sentence-transformer embeddings + cosine similarity.")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    st.caption("Optional controls narrow results after semantic matching.")
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
    max_time = None
    if use_time_limit:
        max_time = st.slider(
            "Max cooking time (minutes)",
            min_value=5,
            max_value=180,
            value=60,
            step=5,
        )

    use_ingredient_limit = st.checkbox("Apply max ingredients filter", value=False)
    max_ingredients = None
    if use_ingredient_limit:
        max_ingredients = st.slider(
            "Max ingredients",
            min_value=2,
            max_value=30,
            value=15,
            step=1,
        )

    top_k = st.slider("Number of results", min_value=3, max_value=20, value=8)
    alpha = st.slider(
        "Image vs text weight (combined search)",
        min_value=0.1,
        max_value=0.9,
        value=0.75,
        step=0.05,
        help="Higher keeps the uploaded dish as the anchor. Lower lets the text steer more.",
    )
    st.markdown("---")
    st.caption(f"Shopping list items: {get_shopping_list_count()}")
    st.caption("Open the Shopping List page to review, check off, or edit items.")

col_text, col_upload = st.columns([2, 1])

with col_text:
    query = st.text_input(
        "Describe what you want to eat",
        placeholder="quick spicy tofu dinner...",
        key="search_query_input",
    )
    _render_search_history()

with col_upload:
    uploaded_file = st.file_uploader(
        "Or upload a dish photo",
        type=["png", "jpg", "jpeg"],
        help="Image search uses SigLIP embeddings; text search stays on SBERT.",
    )

search_clicked = st.button("Search", type="primary", use_container_width=True)
history_search_requested = bool(st.session_state.pop("history_search_requested", False))
run_search = search_clicked or history_search_requested

if run_search:
    svc = get_retrieval_service()
    _reset_feedback()
    request = RetrievalRequest(
        query=query,
        top_k=top_k,
        dietary_filter=diet_options[diet_label],
        max_time_minutes=max_time,
        max_ingredients=max_ingredients,
    )
    has_query = bool(query.strip())
    has_image = uploaded_file is not None
    results_placeholder = st.empty()

    if has_query or has_image:
        with results_placeholder.container():
            _render_skeleton_grid(top_k)

    if has_image and has_query:
        image = Image.open(uploaded_file)
        st.image(image, caption="Searching with image + text…", width=220)
        results = svc.search_combined(query, image, request, alpha=alpha)
        search_mode = "image+text"
    elif has_image:
        image = Image.open(uploaded_file)
        st.image(image, caption="Searching by this image…", width=220)
        results = svc.search_by_image(image, request)
        search_mode = "image"
    elif has_query:
        st.session_state["feedback_query_vec"] = svc.encode_text_query(query)
        st.session_state["feedback_active_request"] = request
        results = svc.search(request)
        search_mode = "text"
    else:
        st.warning("Please enter a search query or upload an image.")
        results = None
        search_mode = ""

    results_placeholder.empty()

    if isinstance(results, pd.DataFrame):
        st.session_state["search_results_df"] = results.copy()
        st.session_state["last_query"] = query.strip()
        st.session_state["last_search_mode"] = search_mode
        _add_to_history(query)

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
        show_feedback = (
            search_mode == "text"
            and st.session_state.get("feedback_query_vec") is not None
        )

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
            recipe_id = str(row_dict.get("recipe_id") or row_dict.get("id") or rank)
            render_recipe_card(
                row_dict,
                rank=rank,
                display_mode=display_mode,
                on_add_to_shopping_list=_add_recipe_to_shopping_list,
                on_negative_feedback=_on_negative_feedback if show_feedback else None,
                feedback_key=f"feedback_{recipe_id}_{rank}" if show_feedback else None,
                widget_key_prefix=f"results_{search_mode or 'text'}",
            )
else:
    _render_landing_state()
