"""UI helpers for reusable filter widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_basic_filters() -> dict[str, Any]:
    """Render common recipe filter widgets and return the selected values."""
    dietary = st.selectbox("Diet", ["Any", "Vegetarian", "Vegan", "Gluten-free"])
    max_time = st.slider("Max time", 5, 180, 45)
    calories = st.slider("Max calories", 100, 10000, 500)
    fat = st.slider("Max total fat (g)", 0, 17183, 20)
    sugar = st.slider("Max sugar (g)", 0, 362729, 20)
    sodium = st.slider("Max sodium (mg)", 0, 29338, 500)
    protein = st.slider("Max protein (g)", 0, 6552, 20)
    saturated_fat = st.slider("Max saturated fat (g)", 0, 10395, 10)
    carbonhydrates = st.slider("Max carbohydrates (g)", 0, 30698, 50)
    total_ratings = st.slider("Average Rating", 0, 5, 4, step = 0.1)

    return {
        "dietary": dietary,
        "max_time": max_time,
        "calories": calories,
        "fat": fat,
        "sugar": sugar,
        "sodium": sodium,
        "protein": protein,
        "saturated_fat": saturated_fat,
        "carbonhydrates": carbonhydrates,
        "total_ratings": total_ratings,
    }
