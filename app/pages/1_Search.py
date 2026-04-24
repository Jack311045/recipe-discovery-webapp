"""Semantic search page — wired to RetrievalService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st

from app.service_loader import get_retrieval_service
from app.components.recipe_cards import render_recipe_card
from recipe_discovery.retrieval.service import RetrievalRequest

st.set_page_config(page_title="Search Recipes", layout="wide")
st.title("🔍 Search Recipes")
st.caption("Semantic retrieval powered by sentence-transformer embeddings + cosine similarity.")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    diet_options = {"Any": None, "Vegetarian": "vegetarian", "Vegan": "vegan", "Gluten-free": "gluten-free"}
    diet_label = st.selectbox("Dietary preference", list(diet_options.keys()))
    max_time = st.slider("Max cooking time (minutes)", min_value=5, max_value=180, value=60, step=5)
    max_ingredients = st.slider("Max ingredients", min_value=2, max_value=30, value=15, step=1)
    top_k = st.slider("Number of results", min_value=3, max_value=20, value=8)

query = st.text_input(
    "Describe what you want to eat",
    placeholder="quick spicy tofu dinner…",
)

if st.button("Search", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching…"):
            svc = get_retrieval_service()
            request = RetrievalRequest(
                query=query,
                top_k=top_k,
                dietary_filter=diet_options[diet_label],
                max_time_minutes=max_time,
                max_ingredients=max_ingredients,
            )
            results = svc.search(request)

        if results.empty:
            st.info("No recipes matched your filters. Try relaxing the constraints.")
        else:
            st.success(f"Found **{len(results)}** recipes for: *{query}*")
            has_proj = "x_proj" in results.columns
            if has_proj:
                st.caption("📍 PCA coordinates are attached — check the Embedding Map page to see where these land!")

            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                render_recipe_card(row.to_dict(), rank=rank)
