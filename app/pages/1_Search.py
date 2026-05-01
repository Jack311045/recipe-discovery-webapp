"""Semantic search page - wired to RetrievalService."""

from __future__ import annotations

import json
import random
import sys
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from app.components.theme import apply_restaurant_menu_theme

apply_restaurant_menu_theme()

import pandas as pd
import streamlit.components.v1 as components
from PIL import Image

from app.components.search_ui import (
    DEFAULT_SEARCH_RESULT_LIMIT,
    LOAD_MORE_RESULT_INCREMENT,
    SEARCH_PREFETCH_RESULT_LIMIT,
    SORT_OPTIONS,
    build_result_summary,
    get_active_tags,
    infer_tag_columns,
    merge_appended_results,
    sort_results_for_display,
)
from app.components.recipe_cards import parse_ingredients, render_recipe_card
from app.components.search_state import (
    SEARCH_STATE_QUERY_PARAM,
    restore_search_snapshot,
    save_search_snapshot,
)
from app.components.shopping_list import (
    add_ingredients_to_shopping_list,
    ensure_shopping_list_state,
    get_shopping_list_count,
)
from app.cluster_filter import (
    build_dropdown_options,
    cluster_artifacts_available,
    cluster_filter_top_k_multiplier,
    filter_results_by_cluster,
    load_cluster_assignments,
    load_cluster_name_map,
)
from app.service_loader import get_retrieval_service

LANDING_QUERIES = [
    "quick weeknight dinner",
    "healthy breakfast ideas",
    "easy comfort food",
    "simple vegetarian recipes",
]


@st.cache_resource(show_spinner=False)
def _get_prefetch_executor() -> ThreadPoolExecutor:
    """Small background worker pool for warming hidden result caches."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-prefetch")


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
        "search_results_pool_df": None,
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
        "feedback_embedding_space": "text",
        "upload_widget_version": 0,
        "search_result_limit": DEFAULT_SEARCH_RESULT_LIMIT,
        "active_cluster_id": None,
        "last_alpha": 0.75,
        "last_search_image": None,
        "prefetch_future": None,
        "prefetch_signature": "",
        "prefetch_error": "",
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
    st.session_state["feedback_embedding_space"] = "text"


def _reset_prefetch_state() -> None:
    """Clear any in-flight hidden-cache prefetch for a new search context."""
    future = st.session_state.get("prefetch_future")
    if isinstance(future, Future) and not future.done():
        future.cancel()
    st.session_state["prefetch_future"] = None
    st.session_state["prefetch_signature"] = ""
    st.session_state["prefetch_error"] = ""


def _get_query_param(name: str) -> str:
    """Read a single query parameter across Streamlit versions."""
    if hasattr(st, "query_params"):
        raw_value = st.query_params.get(name, "")
        if isinstance(raw_value, list):
            return str(raw_value[0]) if raw_value else ""
        return str(raw_value or "")

    get_query_params = getattr(st, "experimental_get_query_params", None)
    if get_query_params is None:
        return ""

    raw_value = get_query_params().get(name, [])
    if isinstance(raw_value, list):
        return str(raw_value[0]) if raw_value else ""
    return str(raw_value or "")


def _restore_cached_search_from_query() -> None:
    """Restore search results cached before cross-page navigation."""
    token = _get_query_param(SEARCH_STATE_QUERY_PARAM)
    if token:
        restore_search_snapshot(token, st.session_state)


def _save_current_search_snapshot() -> None:
    """Persist current search state so raw page navigation can restore it."""
    if isinstance(st.session_state.get("search_results_df"), pd.DataFrame):
        save_search_snapshot(st.session_state)


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


def _submit_search_from_text() -> None:
    """Submit the current text query when the input commits with Enter."""
    query = str(st.session_state.get("search_query_input", "")).strip()
    if query:
        st.session_state["history_search_requested"] = True


def _attach_search_history_to_input() -> None:
    """Attach recent searches to the native search input as datalist suggestions."""
    history = st.session_state["search_history"]
    if not history:
        return

    history_json = json.dumps(history)
    components.html(
        f"""
        <script>
        (() => {{
            const history = {history_json};
            const listId = "recipe-search-history-list";
            const attach = () => {{
                const doc = window.parent.document;
                const inputs = Array.from(doc.querySelectorAll("input"));
                const input = inputs.find((node) =>
                    (node.getAttribute("aria-label") || "").toLowerCase().includes("search recipes") ||
                    (node.getAttribute("placeholder") || "").toLowerCase().includes("recipe name") ||
                    node.getAttribute("aria-label") === "Search"
                );
                if (!input) {{
                    window.setTimeout(attach, 80);
                    return;
                }}

                let list = doc.getElementById(listId);
                if (!list) {{
                    list = doc.createElement("datalist");
                    list.id = listId;
                    doc.body.appendChild(list);
                }}

                list.innerHTML = "";
                history.forEach((query) => {{
                    const option = doc.createElement("option");
                    option.value = query;
                    list.appendChild(option);
                }});

                input.setAttribute("list", listId);
                input.setAttribute("autocomplete", "off");
            }};
            attach();
        }})();
        </script>
        """,
        height=0,
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


def _clear_uploaded_image() -> None:
    """Reset the file uploader by changing its widget key."""
    st.session_state["upload_widget_version"] += 1


def _open_uploaded_image(uploaded_file) -> Image.Image:
    """Open an uploaded image without leaving the file pointer consumed."""
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).copy()
    uploaded_file.seek(0)
    return image


def _effective_top_k(
    display_limit: int,
    cluster_id: int | None,
    cluster_assignments_df: pd.DataFrame | None,
) -> int:
    """Return the retrieval pool size needed for the visible result limit."""
    if cluster_id is None or cluster_assignments_df is None:
        return int(display_limit)

    multiplier = cluster_filter_top_k_multiplier(cluster_id, cluster_assignments_df)
    return int(display_limit * multiplier)


def _apply_cluster_filter_and_trim(
    results: pd.DataFrame,
    *,
    display_limit: int,
    cluster_id: int | None,
    cluster_assignments_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply the optional cluster filter and trim back to the visible limit."""
    if cluster_id is not None and cluster_assignments_df is not None:
        results = filter_results_by_cluster(results, cluster_id, cluster_assignments_df)

    if len(results) > display_limit:
        results = results.iloc[:display_limit].reset_index(drop=True)

    return results.reset_index(drop=True)


def _visible_request_for_limit(
    request,
    display_limit: int,
    cluster_id: int | None,
    cluster_assignments_df: pd.DataFrame | None,
):
    """Return a request sized for the rows currently visible to the user."""
    return replace(
        request,
        top_k=_effective_top_k(display_limit, cluster_id, cluster_assignments_df),
    )


def _filter_excluded_results(results: pd.DataFrame) -> pd.DataFrame:
    """Remove recipes dismissed with negative feedback from a cached pool."""
    excluded = st.session_state.get("feedback_excluded_ids") or set()
    if results.empty or not excluded or "recipe_id" not in results.columns:
        return results

    keep = ~results["recipe_id"].astype(str).isin({str(item) for item in excluded})
    return results.loc[keep].reset_index(drop=True)


def _prefetch_signature(
    *,
    search_mode: str,
    search_query: str,
    request,
    target_limit: int,
    cluster_id: int | None,
    alpha: float,
    image: Image.Image | None,
) -> str:
    """Build a stable signature so stale background results are ignored."""
    return json.dumps(
        {
            "mode": search_mode,
            "query": search_query,
            "diet": getattr(request, "dietary_filter", None),
            "time": getattr(request, "max_time_minutes", None),
            "ingredients": getattr(request, "max_ingredients", None),
            "target_limit": target_limit,
            "cluster_id": int(cluster_id) if cluster_id is not None else None,
            "alpha": round(float(alpha), 3),
            "image_id": id(image) if image is not None else None,
        },
        sort_keys=True,
    )


def _prefetch_results_worker(
    *,
    svc,
    search_mode: str,
    search_query: str,
    request,
    target_limit: int,
    cluster_id: int | None,
    cluster_assignments_df: pd.DataFrame | None,
    alpha: float,
    image: Image.Image | None,
) -> pd.DataFrame:
    """Fetch a larger hidden result pool without touching Streamlit state."""
    effective_top_k = _effective_top_k(target_limit, cluster_id, cluster_assignments_df)
    prefetch_request = replace(request, top_k=effective_top_k)

    if search_mode == "image+text":
        if image is None:
            return pd.DataFrame()
        results = svc.search_combined(search_query, image, prefetch_request, alpha=alpha)
    elif search_mode == "image":
        if image is None:
            return pd.DataFrame()
        results = svc.search_by_image(image, prefetch_request)
    else:
        results = svc.search(prefetch_request)

    return _apply_cluster_filter_and_trim(
        results,
        display_limit=target_limit,
        cluster_id=cluster_id,
        cluster_assignments_df=cluster_assignments_df,
    )


def _harvest_prefetch_if_ready(*, wait_seconds: float = 0.0) -> bool:
    """Move completed background prefetch results into the hidden result pool."""
    future = st.session_state.get("prefetch_future")
    if not isinstance(future, Future):
        return False
    if not future.done() and wait_seconds <= 0:
        return False

    signature = st.session_state.get("prefetch_signature", "")
    try:
        prefetched = future.result(timeout=wait_seconds)
    except TimeoutError:
        return False
    except Exception as exc:
        st.session_state["prefetch_error"] = str(exc)
        st.session_state["prefetch_future"] = None
        st.session_state["prefetch_signature"] = ""
        return False

    if signature != st.session_state.get("prefetch_signature", ""):
        return False

    existing_pool = st.session_state.get("search_results_pool_df")
    if isinstance(existing_pool, pd.DataFrame) and not existing_pool.empty:
        target_limit = max(len(existing_pool), len(prefetched))
        prefetched = merge_appended_results(
            existing_pool,
            prefetched,
            limit=target_limit,
        )

    st.session_state["search_results_pool_df"] = prefetched.copy()
    st.session_state["prefetch_future"] = None
    st.session_state["prefetch_signature"] = ""
    st.session_state["prefetch_error"] = ""
    _save_current_search_snapshot()
    return True


def _start_prefetch_if_needed(
    *,
    search_mode: str,
    search_query: str,
    cluster_assignments_df: pd.DataFrame | None,
) -> None:
    """Start warming the hidden result pool after visible results are rendered."""
    base_request = st.session_state.get("feedback_active_request")
    if base_request is None or search_mode not in {"text", "image", "image+text"}:
        return

    future = st.session_state.get("prefetch_future")
    if isinstance(future, Future) and not future.done():
        return

    current_limit = int(
        st.session_state.get("search_result_limit") or DEFAULT_SEARCH_RESULT_LIMIT
    )
    target_limit = max(
        SEARCH_PREFETCH_RESULT_LIMIT,
        current_limit + LOAD_MORE_RESULT_INCREMENT * 2,
    )
    pool = st.session_state.get("search_results_pool_df")
    if isinstance(pool, pd.DataFrame) and len(pool) >= target_limit:
        return

    active_cluster_id = st.session_state.get("active_cluster_id")
    active_cluster_assignments_df = cluster_assignments_df
    if active_cluster_id is not None and (
        active_cluster_assignments_df is None or active_cluster_assignments_df.empty
    ):
        active_cluster_assignments_df = load_cluster_assignments()

    image = st.session_state.get("last_search_image")
    image_for_worker = image.copy() if isinstance(image, Image.Image) else None
    alpha = float(st.session_state.get("last_alpha", 0.75))
    signature = _prefetch_signature(
        search_mode=search_mode,
        search_query=search_query,
        request=base_request,
        target_limit=target_limit,
        cluster_id=active_cluster_id,
        alpha=alpha,
        image=image,
    )
    if st.session_state.get("prefetch_signature") == signature:
        return

    svc = get_retrieval_service()
    future = _get_prefetch_executor().submit(
        _prefetch_results_worker,
        svc=svc,
        search_mode=search_mode,
        search_query=search_query,
        request=base_request,
        target_limit=target_limit,
        cluster_id=active_cluster_id,
        cluster_assignments_df=active_cluster_assignments_df,
        alpha=alpha,
        image=image_for_worker,
    )
    st.session_state["prefetch_future"] = future
    st.session_state["prefetch_signature"] = signature
    st.session_state["prefetch_error"] = ""


def _cached_results_for_limit(
    existing_results: pd.DataFrame,
    *,
    next_limit: int,
) -> pd.DataFrame | None:
    """Return visible rows from the cached result pool when enough are available."""
    pool = st.session_state.get("search_results_pool_df")
    if not isinstance(pool, pd.DataFrame) or pool.empty:
        return None

    pool = _filter_excluded_results(pool)
    if len(pool) <= len(existing_results):
        return None

    cached_limit = min(next_limit, len(pool))
    cached_results = merge_appended_results(
        existing_results,
        pool,
        limit=cached_limit,
    )
    if len(cached_results) <= len(existing_results):
        return None
    return cached_results


def _load_more_results(
    *,
    search_mode: str,
    search_query: str,
    existing_results: pd.DataFrame,
    cluster_assignments_df: pd.DataFrame | None,
) -> None:
    """Fetch the next small batch for the current query and append it."""
    _harvest_prefetch_if_ready(wait_seconds=0.25)
    base_request = st.session_state.get("feedback_active_request")
    if base_request is None:
        st.info("Run a search first, then load more matches from that query.")
        return

    current_limit = int(
        st.session_state.get("search_result_limit") or DEFAULT_SEARCH_RESULT_LIMIT
    )
    next_limit = current_limit + LOAD_MORE_RESULT_INCREMENT
    active_cluster_id = st.session_state.get("active_cluster_id")
    active_cluster_assignments_df = cluster_assignments_df
    if active_cluster_id is not None and (
        active_cluster_assignments_df is None or active_cluster_assignments_df.empty
    ):
        active_cluster_assignments_df = load_cluster_assignments()

    cached_results = _cached_results_for_limit(
        existing_results,
        next_limit=next_limit,
    )
    if cached_results is not None:
        st.session_state["search_result_limit"] = len(cached_results)
        st.session_state["feedback_active_request"] = _visible_request_for_limit(
            base_request,
            len(cached_results),
            active_cluster_id,
            active_cluster_assignments_df,
        )
        st.session_state["search_results_df"] = cached_results.copy()
        _save_current_search_snapshot()
        st.rerun()

    future = st.session_state.get("prefetch_future")
    if isinstance(future, Future) and not future.done():
        st.info("Preparing more recipes now. Try again in a moment.")
        return

    cache_limit = next_limit + LOAD_MORE_RESULT_INCREMENT * 2
    effective_top_k = _effective_top_k(
        cache_limit,
        active_cluster_id,
        active_cluster_assignments_df,
    )
    expanded_request = replace(base_request, top_k=effective_top_k)
    svc = get_retrieval_service()

    with st.spinner("Finding a few more recipes..."):
        if search_mode == "image+text":
            image = st.session_state.get("last_search_image")
            if image is None:
                st.warning("Upload the image again to load more image-based matches.")
                return
            expanded_results = svc.search_combined(
                search_query,
                image,
                expanded_request,
                alpha=float(st.session_state.get("last_alpha", 0.75)),
            )
        elif search_mode == "image":
            image = st.session_state.get("last_search_image")
            if image is None:
                st.warning("Upload the image again to load more image-based matches.")
                return
            expanded_results = svc.search_by_image(image, expanded_request)
        else:
            expanded_results = svc.search(expanded_request)

    expanded_results = _apply_cluster_filter_and_trim(
        expanded_results,
        display_limit=cache_limit,
        cluster_id=active_cluster_id,
        cluster_assignments_df=active_cluster_assignments_df,
    )
    expanded_results = _filter_excluded_results(expanded_results)
    pool = st.session_state.get("search_results_pool_df")
    if isinstance(pool, pd.DataFrame) and not pool.empty:
        expanded_results = merge_appended_results(
            pool,
            expanded_results,
            limit=cache_limit,
        )

    combined_results = merge_appended_results(
        existing_results,
        expanded_results,
        limit=next_limit,
    )
    if len(combined_results) <= len(existing_results):
        st.info("No more recipes matched this search.")
        return

    visible_count = len(combined_results)
    st.session_state["search_result_limit"] = visible_count
    st.session_state["feedback_active_request"] = _visible_request_for_limit(
        base_request,
        visible_count,
        active_cluster_id,
        active_cluster_assignments_df,
    )
    st.session_state["search_results_df"] = combined_results.copy()
    st.session_state["search_results_pool_df"] = expanded_results.copy()
    _save_current_search_snapshot()
    st.rerun()


def _load_landing_results() -> pd.DataFrame:
    """Load and cache first-run landing results for this session."""
    if st.session_state["landing_results_df"] is None:
        from recipe_discovery.retrieval.service import RetrievalRequest

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
    _render_result_grid(
        landing_df,
        tag_columns=tag_columns,
        max_tags=8,
        search_mode="landing",
        show_feedback=True,
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
    st.session_state["shopping_list_notice"] = message
    st.rerun()


def _on_negative_feedback(recipe_id: str) -> None:
    """Apply negative feedback to the active text-search results or landing results."""
    excluded = st.session_state["feedback_excluded_ids"]
    excluded.add(str(recipe_id))

    current = st.session_state.get("search_results_df")
    
    # If no search is active, user is dismissing a landing page card
    if current is None:
        landing_df = st.session_state.get("landing_results_df")
        if isinstance(landing_df, pd.DataFrame) and "recipe_id" in landing_df.columns:
            keep = landing_df["recipe_id"].astype(str) != str(recipe_id)
            st.session_state["landing_results_df"] = landing_df.loc[keep].reset_index(drop=True)
        return

    query_vec = st.session_state.get("feedback_query_vec")
    request = st.session_state.get("feedback_active_request")
    if query_vec is None or request is None:
        if isinstance(current, pd.DataFrame) and "recipe_id" in current.columns:
            keep = current["recipe_id"].astype(str) != str(recipe_id)
            st.session_state["search_results_df"] = current.loc[keep].reset_index(drop=True)
            _save_current_search_snapshot()
        return

    svc = get_retrieval_service()
    results = svc.search_with_negative_feedback(
        request,
        query_vec=query_vec,
        negative_recipe_ids=excluded,
        alpha=0.3,
        embedding_space=str(st.session_state.get("feedback_embedding_space") or "text"),
    )
    st.session_state["search_results_df"] = results.copy()
    _save_current_search_snapshot()


def _render_result_grid(
    display_df: pd.DataFrame,
    *,
    tag_columns: list[str],
    max_tags: int,
    search_mode: str,
    show_feedback: bool,
) -> None:
    """Render detailed recipe cards in a two-column grid."""
    columns_per_row = 2
    for start in range(0, len(display_df), columns_per_row):
        cols = st.columns(columns_per_row, gap="medium")
        chunk = display_df.iloc[start : start + columns_per_row]
        for offset, (_, row) in enumerate(chunk.iterrows()):
            rank = start + offset + 1
            row_dict = row.to_dict()
            row_dict["_active_tags"] = get_active_tags(
                row_dict,
                tag_columns,
                max_tags=max_tags,
            )
            recipe_id = str(row_dict.get("recipe_id") or row_dict.get("id") or rank)
            with cols[offset]:
                render_recipe_card(
                    row_dict,
                    rank=rank,
                    display_mode="Detailed",
                    on_add_to_shopping_list=_add_recipe_to_shopping_list,
                    on_negative_feedback=_on_negative_feedback if show_feedback else None,
                    feedback_key=f"feedback_{recipe_id}_{rank}" if show_feedback else None,
                    widget_key_prefix=f"results_{search_mode or 'text'}",
                )


_initialize_session_state()
_restore_cached_search_from_query()
shopping_list_notice = st.session_state.pop("shopping_list_notice", None)
if shopping_list_notice:
    if hasattr(st, "toast"):
        st.toast(shopping_list_notice)
    else:
        st.success(shopping_list_notice)

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
    .upload-preview-row {
        display: grid;
        grid-template-columns: 42px 1fr;
        gap: 0.35rem;
        align-items: center;
        margin-top: 0.18rem;
        min-height: 2.1rem;
    }
    .upload-empty {
        min-height: 2.2rem;
        border: 1px dashed rgba(183, 159, 115, 0.8);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6c5848;
        font-size: 0.92rem;
        text-align: center;
        padding: 0.35rem 0.55rem;
        background: rgba(255, 253, 247, 0.68);
    }
    .upload-thumb-note {
        font-size: 0.9rem;
        color: #6c5848;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="menu-page-header">
      <p class="menu-page-kicker">Today's Discovery Menu</p>
      <h1 class="menu-page-title">Search Recipes</h1>
      <p class="menu-page-subtitle">Search by recipe name, ingredients, cuisine style, tags, or meal ideas. Photo upload is optional and can be combined with text search.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

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

    # Cluster filter (only rendered when k-means artifact is available)
    selected_cluster_id: int | None = None
    cluster_assignments_df: pd.DataFrame | None = None
    if cluster_artifacts_available():
        cluster_assignments_df = load_cluster_assignments()
        if cluster_assignments_df is not None and not cluster_assignments_df.empty:
            cluster_name_map = load_cluster_name_map(cluster_assignments_df)
            cluster_options = build_dropdown_options(cluster_assignments_df, cluster_name_map)
            cluster_label_to_id = dict(cluster_options)
            cluster_label = st.selectbox(
                "Limit to cluster",
                options=[label for label, _ in cluster_options],
                index=0,
                help=(
                    "Restrict search to recipes within a specific semantic cluster. "
                    "Use this to focus, e.g., on pasta-style recipes when searching "
                    "for 'summer dinner'."
                ),
            )
            selected_cluster_id = cluster_label_to_id[cluster_label]

    alpha = st.slider(
        "Image vs text weight (combined search)",
        min_value=0.1,
        max_value=0.9,
        value=0.75,
        step=0.05,
        help="Higher keeps the uploaded dish as the anchor. Lower lets the text steer more.",
    )

with st.container(border=True):
    st.markdown("#### Find your next recipe")
    st.markdown(
        "<p class='search-guidance-text'>Type recipe names, ingredients, cuisine styles, tags, or meal ideas. Uploading a photo is optional.</p>",
        unsafe_allow_html=True,
    )

    col_text, col_search, col_upload = st.columns([4.6, 1.2, 2.2], gap="medium")

    with col_text:
        query = st.text_input(
            "Search recipes",
            placeholder="Try: creamy mushroom pasta, spicy tofu bowl, gluten-free brunch, Italian weeknight dinner...",
            key="search_query_input",
            on_change=_submit_search_from_text,
            autocomplete="off",
            help="You can type recipe names, ingredients, cuisine styles, tags, or meal ideas.",
        )
        _attach_search_history_to_input()

    with col_search:
        st.markdown("<div style='height: 1.95rem;'></div>", unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    with col_upload:
        upload_key = f"photo_upload_{st.session_state['upload_widget_version']}"
        uploaded_file = st.file_uploader(
            "Optional photo upload",
            type=["png", "jpg", "jpeg"],
            help="Optional: upload a dish photo for image-only search, or combine it with text search.",
            key=upload_key,
        )
        if uploaded_file is None:
            st.markdown(
                "<div class='upload-empty'>Optional: add a dish photo for image-based search.</div>",
                unsafe_allow_html=True,
            )
        else:
            preview_cols = st.columns([0.65, 1.2], gap="small")
            with preview_cols[0]:
                st.image(_open_uploaded_image(uploaded_file), width=42)
            with preview_cols[1]:
                st.markdown(
                    "<div class='upload-thumb-note'>Image ready for search</div>",
                    unsafe_allow_html=True,
                )
                st.button(
                    "Remove",
                    key="clear_uploaded_image",
                    use_container_width=True,
                    on_click=_clear_uploaded_image,
                )

    st.markdown("##### Display options")
    col_sort, col_tags = st.columns([1.35, 1.8], gap="small")
    with col_sort:
        sort_mode = st.selectbox(
            "Sort displayed results",
            SORT_OPTIONS,
            index=0,
            help="Display-only ordering. Retrieval and backend ranking are unchanged.",
        )
    with col_tags:
        max_tags = st.slider(
            "Maximum tag chips shown per recipe",
            min_value=3,
            max_value=12,
            value=8,
            help="Tag chips are visual only and do not affect ranking/filter behavior.",
        )
history_search_requested = bool(st.session_state.pop("history_search_requested", False))
run_search = search_clicked or history_search_requested

if run_search:
    from recipe_discovery.retrieval.service import RetrievalRequest

    svc = get_retrieval_service()
    _reset_feedback()
    _reset_prefetch_state()
    display_limit = DEFAULT_SEARCH_RESULT_LIMIT
    st.session_state["search_result_limit"] = display_limit
    st.session_state["active_cluster_id"] = selected_cluster_id
    st.session_state["last_alpha"] = alpha

    # Keep the first search small; a background prefetch warms hidden rows after
    # these visible results have already rendered.
    effective_top_k = _effective_top_k(
        display_limit,
        selected_cluster_id,
        cluster_assignments_df,
    )

    request = RetrievalRequest(
        query=query,
        top_k=effective_top_k,
        dietary_filter=diet_options[diet_label],
        max_time_minutes=max_time,
        max_ingredients=max_ingredients,
    )
    has_query = bool(query.strip())
    has_image = uploaded_file is not None
    results_placeholder = st.empty()

    if has_query or has_image:
        with results_placeholder.container():
            _render_skeleton_grid(display_limit)

    if has_image and has_query:
        image = _open_uploaded_image(uploaded_file)
        st.session_state["last_search_image"] = image.copy()
        try:
            results = svc.search_combined(query, image, request, alpha=alpha)
            st.session_state["feedback_query_vec"] = svc.encode_combined_query(
                query,
                image,
                alpha=alpha,
            )
            st.session_state["feedback_active_request"] = request
            st.session_state["feedback_embedding_space"] = "siglip"
            search_mode = "image+text"
        except RuntimeError as exc:
            st.warning(f"Image search is unavailable: {exc}")
            results = None
            search_mode = ""
    elif has_image:
        image = _open_uploaded_image(uploaded_file)
        st.session_state["last_search_image"] = image.copy()
        try:
            results = svc.search_by_image(image, request)
            st.session_state["feedback_query_vec"] = svc.encode_image_query(image)
            st.session_state["feedback_active_request"] = request
            st.session_state["feedback_embedding_space"] = "siglip"
            search_mode = "image"
        except RuntimeError as exc:
            st.warning(f"Image search is unavailable: {exc}")
            results = None
            search_mode = ""
    elif has_query:
        st.session_state["last_search_image"] = None
        st.session_state["feedback_query_vec"] = svc.encode_text_query(query)
        st.session_state["feedback_active_request"] = request
        st.session_state["feedback_embedding_space"] = "text"
        results = svc.search(request)
        search_mode = "text"
    else:
        st.warning("Please enter a search query or upload an image.")
        st.session_state["last_search_image"] = None
        results = None
        search_mode = ""

    results_placeholder.empty()

    # Apply cluster filter (if active) and trim to the visible result limit.
    if isinstance(results, pd.DataFrame):
        results = _apply_cluster_filter_and_trim(
            results,
            display_limit=display_limit,
            cluster_id=selected_cluster_id,
            cluster_assignments_df=cluster_assignments_df,
        )

    if isinstance(results, pd.DataFrame):
        st.session_state["search_results_pool_df"] = results.copy()
        st.session_state["search_results_df"] = results.copy()
        st.session_state["last_query"] = query.strip()
        st.session_state["last_search_mode"] = search_mode
        _add_to_history(query)
        _save_current_search_snapshot()
        st.rerun()

results_df = st.session_state.get("search_results_df")
search_query = st.session_state.get("last_query", "")
search_mode = st.session_state.get("last_search_mode", "")
_harvest_prefetch_if_ready()

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

        display_df = sort_results_for_display(results_df, sort_mode)
        tag_columns = infer_tag_columns(display_df)
        show_feedback = st.session_state.get("feedback_query_vec") is not None

        has_proj = "x_proj" in display_df.columns
        if has_proj:
            st.caption(
                "📍 PCA coordinates are attached - check the Embedding Map page "
                "to see where these land!"
            )

        _render_result_grid(
            display_df,
            tag_columns=tag_columns,
            max_tags=max_tags,
            search_mode=search_mode,
            show_feedback=show_feedback,
        )

        if (
            search_mode in {"text", "image", "image+text"}
            and st.session_state.get("feedback_active_request") is not None
        ):
            load_more_cols = st.columns([1.2, 1, 1.2])
            with load_more_cols[1]:
                if st.button(
                    "Get a few more",
                    key="load_more_results",
                    use_container_width=True,
                ):
                    _load_more_results(
                        search_mode=search_mode,
                        search_query=search_query,
                        existing_results=results_df,
                        cluster_assignments_df=cluster_assignments_df,
                    )
            future = st.session_state.get("prefetch_future")
            if isinstance(future, Future) and not future.done():
                st.caption("Preparing more matches in the background.")

        _start_prefetch_if_needed(
            search_mode=search_mode,
            search_query=search_query,
            cluster_assignments_df=cluster_assignments_df,
        )
else:
    _render_landing_state()
