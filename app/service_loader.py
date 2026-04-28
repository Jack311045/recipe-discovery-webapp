"""Cached backend service loader shared across Streamlit pages."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure recipe_discovery is importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


@st.cache_resource(show_spinner="Loading retrieval service…")
def get_retrieval_service():
    """Load RetrievalService once and cache it for the entire session."""
    from recipe_discovery.retrieval.service import RetrievalService

    svc = RetrievalService()
    svc.load()
    return svc
