"""UI helpers for reusable filter widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_basic_filters() -> dict[str, Any]:
    """Render common recipe filter widgets and return the selected values."""
    dietary = st.selectbox("Diet", ["Any", "Vegetarian", "Vegan", "Gluten-free"])
    max_time = st.slider("Max time", 5, 180, 45)
    return {
        "dietary": dietary,
        "max_time": max_time,
    }
