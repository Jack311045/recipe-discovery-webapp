"""Cached backend service loader shared across Streamlit pages."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure recipe_discovery is importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


@st.cache_resource(show_spinner=False)
def _build_retrieval_service():
    """Build RetrievalService once and cache it across reruns."""
    from recipe_discovery.retrieval.service import RetrievalService

    svc = RetrievalService()
    svc.load()
    return svc


def get_retrieval_service():
    """Return the cached RetrievalService, showing a spinner only on first load."""
    if "retrieval_service" not in st.session_state:
        with st.spinner("Loading retrieval service..."):
            st.session_state["retrieval_service"] = _build_retrieval_service()
    return st.session_state["retrieval_service"]
