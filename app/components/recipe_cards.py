"""Helpers for rendering recipe results."""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st


def render_recipe_card(recipe: Mapping[str, object], rank: int | None = None) -> None:
    """Render a recipe result card with key metadata."""
    title = str(recipe.get("name", recipe.get("title", "Untitled recipe")))
    score = recipe.get("similarity_score")
    minutes = recipe.get("minutes")
    n_ingredients = recipe.get("n_ingredients")
    n_steps = recipe.get("n_steps")
    description = recipe.get("description", "")
    x_proj = recipe.get("x_proj")
    y_proj = recipe.get("y_proj")

    with st.container(border=True):
        header = f"**{rank}. {title}**" if rank is not None else f"**{title}**"
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(header)
        with cols[1]:
            if score is not None:
                st.metric("Match", f"{float(score):.2%}")

        meta_parts = []
        if minutes is not None:
            meta_parts.append(f"\u23f1 {int(minutes)} min")
        if n_ingredients is not None:
            meta_parts.append(f"\U0001f9c2 {int(n_ingredients)} ingredients")
        if n_steps is not None:
            meta_parts.append(f"\U0001f4cb {int(n_steps)} steps")
        if meta_parts:
            st.caption("  \u00b7  ".join(meta_parts))

        if description:
            st.write(str(description)[:300] + ("\u2026" if len(str(description)) > 300 else ""))

        if x_proj is not None and y_proj is not None:
            st.caption(f"\U0001f4cd PCA coords: ({float(x_proj):.3f}, {float(y_proj):.3f})")
