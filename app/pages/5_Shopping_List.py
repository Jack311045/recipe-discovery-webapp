"""Shopping list page for collecting recipe ingredients across searches."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.components.search_state import SEARCH_STATE_QUERY_PARAM, restore_search_snapshot
from app.components.theme import apply_restaurant_menu_theme
from app.components.shopping_list import (
    add_manual_item_to_shopping_list,
    clear_shopping_list,
    ensure_shopping_list_state,
    get_checked_item_count,
    get_grouped_shopping_items,
    get_shopping_list_count,
    remove_checked_items,
    remove_shopping_item,
    set_item_checked,
)

st.set_page_config(page_title="Shopping List", layout="wide")
apply_restaurant_menu_theme()
st.markdown(
    """
    <style>
    .shopping-input-note {
        margin: 0.15rem 0 0.6rem 0;
        font-size: 1rem;
        color: #5b483c;
    }
    .shopping-section-heading {
        margin: 0.85rem 0 0.35rem 0;
        font-size: 1.3rem;
        color: #5b4434;
        font-family: "Cormorant Garamond", "Playfair Display", serif;
    }
    .shopping-empty-state {
        margin-top: 0.55rem;
        padding: 0.9rem 1rem;
        border: 1px dashed rgba(183, 159, 115, 0.9);
        border-radius: 0.75rem;
        background: rgba(255, 252, 245, 0.92);
        color: #5a473b;
        font-size: 1.02rem;
        line-height: 1.5;
    }
    div[data-testid="stCheckbox"] label p {
        font-size: 1.05rem !important;
        line-height: 1.45 !important;
        color: #3f322c !important;
    }
    .shopping-source-caption {
        font-size: 0.95rem;
        color: #665244;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _item_widget_suffix(normalized_name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", normalized_name).strip("_")
    return clean or "item"


def _format_sources(sources: object) -> str:
    if not isinstance(sources, list) or not sources:
        return ""
    source_strings = [str(source).strip() for source in sources if str(source).strip()]
    if not source_strings:
        return ""
    if len(source_strings) <= 3:
        return ", ".join(source_strings)
    return f"{', '.join(source_strings[:3])} (+{len(source_strings) - 3} more)"


def _clear_checkbox_widget_state() -> None:
    keys_to_remove = [
        key
        for key in st.session_state.keys()
        if key.startswith("shopping_checked_")
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)


def _clear_one_checkbox_widget_state(normalized_name: str) -> None:
    suffix = _item_widget_suffix(normalized_name)
    st.session_state.pop(f"shopping_checked_{suffix}", None)


def _get_search_state_query_token() -> str:
    """Read the cached search-state token from Streamlit query params."""
    return _get_query_param(SEARCH_STATE_QUERY_PARAM)


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


ensure_shopping_list_state()
search_state_token = _get_search_state_query_token()
if search_state_token:
    restore_search_snapshot(search_state_token, st.session_state)

st.markdown(
    """
    <section class="menu-page-header">
      <p class="menu-page-kicker">Mise en Place</p>
      <h1 class="menu-page-title">Shopping List</h1>
      <p class="menu-page-subtitle">Collect ingredients from recipe cards, check off items while shopping, and add anything missing manually.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

shopping_list_notice = st.session_state.pop("shopping_list_notice", None)
if shopping_list_notice:
    st.success(shopping_list_notice)

st.markdown(
    "<p class='shopping-input-note'>Add ingredients manually when they are not already pulled from recipe cards.</p>",
    unsafe_allow_html=True,
)

with st.form("shopping_manual_add_form", clear_on_submit=True):
    manual_item = st.text_input(
        "Add a manual shopping item",
        placeholder="e.g., fresh basil, olive oil, chicken thighs, jasmine rice",
        help="Type a single ingredient or grocery item. It will merge with existing duplicates automatically.",
    )
    manual_submitted = st.form_submit_button("Add item")

if manual_submitted:
    if add_manual_item_to_shopping_list(manual_item):
        st.session_state["shopping_list_notice"] = "Added item to shopping list."
        st.rerun()
    else:
        st.warning("Enter a valid item name to add.")

total_items = get_shopping_list_count()
checked_items = get_checked_item_count()
remaining_items = max(total_items - checked_items, 0)

metric_cols = st.columns(3)
with metric_cols[0]:
    st.metric("Items", total_items)
with metric_cols[1]:
    st.metric("Completed", checked_items)
with metric_cols[2]:
    st.metric("Remaining", remaining_items)

if total_items > 0:
    st.progress(checked_items / max(total_items, 1), text="Completion progress")

action_cols = st.columns([2, 2, 3])
with action_cols[0]:
    remove_checked_clicked = st.button("Remove checked", use_container_width=True)
with action_cols[1]:
    clear_all_clicked = st.button("Clear all", use_container_width=True)

if remove_checked_clicked:
    removed_count = remove_checked_items()
    _clear_checkbox_widget_state()
    if removed_count:
        st.success(f"Removed {removed_count} checked item(s).")
    else:
        st.info("No checked items to remove.")
    st.rerun()

if clear_all_clicked:
    clear_shopping_list()
    _clear_checkbox_widget_state()
    st.success("Cleared the shopping list.")
    st.rerun()

if get_shopping_list_count() == 0:
    st.markdown(
        "<div class='shopping-empty-state'>Your shopping list is empty. Add ingredients from search results or enter an item above to get started.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

group_by_category = st.toggle("Group by category", value=True)
grouped_items = get_grouped_shopping_items(group_by_category=group_by_category)

for section_label, items in grouped_items.items():
    if group_by_category:
        st.markdown(f"<h3 class='shopping-section-heading'>{section_label}</h3>", unsafe_allow_html=True)

    for item in items:
        normalized_name = str(item.get("normalized_name", "")).strip()
        if not normalized_name:
            continue

        display_name = str(item.get("display_name", normalized_name)).strip() or normalized_name
        is_checked = bool(item.get("checked", False))
        widget_suffix = _item_widget_suffix(normalized_name)
        checkbox_key = f"shopping_checked_{widget_suffix}"

        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = is_checked

        row_cols = st.columns([6, 3, 1])
        with row_cols[0]:
            checked_now = st.checkbox(display_name, key=checkbox_key)
            if checked_now != is_checked:
                set_item_checked(normalized_name, checked_now)

        with row_cols[1]:
            source_caption = _format_sources(item.get("source_recipes"))
            if source_caption:
                st.markdown(
                    f"<p class='shopping-source-caption'>From: {source_caption}</p>",
                    unsafe_allow_html=True,
                )

        with row_cols[2]:
            remove_clicked = st.button(
                "Remove",
                key=f"shopping_remove_{widget_suffix}",
                use_container_width=True,
            )
            if remove_clicked:
                remove_shopping_item(normalized_name)
                _clear_one_checkbox_widget_state(normalized_name)
                st.rerun()
