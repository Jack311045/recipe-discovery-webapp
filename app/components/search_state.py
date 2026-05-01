"""Cross-page search-state snapshots for Streamlit navigation."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st


SEARCH_STATE_QUERY_PARAM = "search_state"
SEARCH_STATE_TOKEN_KEY = "search_state_token"

_SEARCH_STATE_KEYS = (
    "search_results_df",
    "search_results_pool_df",
    "last_query",
    "last_search_mode",
    "search_query_input",
    "search_history",
    "feedback_query_vec",
    "feedback_excluded_ids",
    "feedback_active_request",
    "feedback_embedding_space",
    "search_result_limit",
    "active_cluster_id",
    "last_alpha",
    "last_search_image",
)


@st.cache_resource
def _search_snapshot_store() -> dict[str, dict[str, Any]]:
    return {}


def _copy_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, set):
        return set(value)
    if isinstance(value, list):
        return list(value)
    copy_value = getattr(value, "copy", None)
    if callable(copy_value):
        try:
            return copy_value()
        except TypeError:
            return value
    return value


def save_search_snapshot(session_state: Any) -> str:
    """Save current search state in a process-local cache and return its token."""
    snapshot: dict[str, Any] = {}
    for key in _SEARCH_STATE_KEYS:
        if key in session_state:
            snapshot[key] = _copy_value(session_state[key])

    token = str(uuid4())
    _search_snapshot_store()[token] = snapshot
    session_state[SEARCH_STATE_TOKEN_KEY] = token
    return token


def restore_search_snapshot(token: str, session_state: Any, *, overwrite: bool = False) -> bool:
    """Restore cached search state into Streamlit session state."""
    token = str(token or "").strip()
    if not token:
        return False

    snapshot = _search_snapshot_store().get(token)
    if not snapshot:
        return False

    if not overwrite and isinstance(session_state.get("search_results_df"), pd.DataFrame):
        session_state[SEARCH_STATE_TOKEN_KEY] = token
        return False

    for key, value in snapshot.items():
        session_state[key] = _copy_value(value)
    session_state[SEARCH_STATE_TOKEN_KEY] = token
    return True
