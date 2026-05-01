"""Cached backend service loader shared across Streamlit pages."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import sys
from threading import Lock
from pathlib import Path

import streamlit as st

# Make sure recipe_discovery is importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_SERVICE_LOCK = Lock()
_SERVICE_SINGLETON = None
_SERVICE_WARMUP_FUTURE: Future | None = None


def _load_retrieval_service():
    """Load RetrievalService without touching Streamlit session state."""
    from recipe_discovery.retrieval.service import RetrievalService

    svc = RetrievalService()
    svc.load()
    return svc


@st.cache_resource(show_spinner=False)
def _get_service_warmup_executor() -> ThreadPoolExecutor:
    """Single worker used to warm the retrieval service after app startup."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-warmup")


@st.cache_resource(show_spinner=False)
def _build_retrieval_service():
    """Build RetrievalService once and cache it across reruns."""
    return _load_retrieval_service()


def start_retrieval_service_warmup() -> None:
    """Begin loading the retrieval service in the background if it is not ready."""
    global _SERVICE_WARMUP_FUTURE

    if "retrieval_service" in st.session_state or _SERVICE_SINGLETON is not None:
        return

    with _SERVICE_LOCK:
        if _SERVICE_WARMUP_FUTURE is None:
            _SERVICE_WARMUP_FUTURE = _get_service_warmup_executor().submit(
                _load_retrieval_service
            )


def _take_warmed_service():
    """Return a completed background service, or None if it is still loading."""
    global _SERVICE_SINGLETON, _SERVICE_WARMUP_FUTURE

    if _SERVICE_SINGLETON is not None:
        return _SERVICE_SINGLETON

    future = _SERVICE_WARMUP_FUTURE
    if future is None or not future.done():
        return None

    try:
        _SERVICE_SINGLETON = future.result()
    finally:
        _SERVICE_WARMUP_FUTURE = None
    return _SERVICE_SINGLETON


def get_retrieval_service():
    """Return the cached RetrievalService, showing a spinner only on first load."""
    global _SERVICE_SINGLETON, _SERVICE_WARMUP_FUTURE

    if "retrieval_service" not in st.session_state:
        warmed = _take_warmed_service()
        if warmed is not None:
            st.session_state["retrieval_service"] = warmed
            return st.session_state["retrieval_service"]

        future = _SERVICE_WARMUP_FUTURE
        if future is not None:
            with st.spinner("Finishing retrieval service load..."):
                _SERVICE_SINGLETON = future.result()
                _SERVICE_WARMUP_FUTURE = None
                st.session_state["retrieval_service"] = _SERVICE_SINGLETON
        else:
            with st.spinner("Loading retrieval service..."):
                st.session_state["retrieval_service"] = _build_retrieval_service()
                _SERVICE_SINGLETON = st.session_state["retrieval_service"]
    return st.session_state["retrieval_service"]
