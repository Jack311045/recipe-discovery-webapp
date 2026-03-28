"""Helpers for rendering recipe results."""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st


def render_recipe_card(recipe: Mapping[str, object]) -> None:
    """Render a minimal recipe result card."""
    title = str(recipe.get("title", "Untitled recipe"))
    with st.container(border=True):
        st.subheader(title)
        st.write(recipe)
