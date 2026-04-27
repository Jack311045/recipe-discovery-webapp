"""Semantic search page — wired to RetrievalService."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from PIL import Image

from app.service_loader import get_retrieval_service
from app.components.recipe_cards import render_recipe_card
from recipe_discovery.retrieval.service import RetrievalRequest

st.set_page_config(page_title="Search Recipes", layout="wide")
st.title("🔍 Search Recipes")
st.caption("Search by text or upload a dish photo for image-based retrieval.")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    diet_options = {"Any": None, "Vegetarian": "vegetarian", "Vegan": "vegan", "Gluten-free": "gluten-free"}
    diet_label = st.selectbox("Dietary preference", list(diet_options.keys()))
    max_time = st.slider("Max cooking time (minutes)", min_value=5, max_value=180, value=60, step=5)
    max_ingredients = st.slider("Max ingredients", min_value=2, max_value=30, value=15, step=1)
    top_k = st.slider("Number of results", min_value=3, max_value=20, value=8)

col_text, col_upload = st.columns([2, 1])

with col_text:
    query = st.text_input(
        "Describe what you want to eat",
        placeholder="quick spicy tofu dinner…",
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Or upload a dish photo",
        type=["png", "jpg", "jpeg"],
        help="Image search uses SigLIP embeddings; text search stays on SBERT.",
    )

if st.button("Search", type="primary", use_container_width=True):
    svc = get_retrieval_service()
    request = RetrievalRequest(
        query=query,
        top_k=top_k,
        dietary_filter=diet_options[diet_label],
        max_time_minutes=max_time,
        max_ingredients=max_ingredients,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Searching by this image…", width=220)
        with st.spinner("Searching with image…"):
            results = svc.search_by_image(image, request)
    elif query.strip():
        with st.spinner("Searching…"):
            results = svc.search(request)
    else:
        st.warning("Please enter a search query or upload an image.")
        results = None

    if results is None:
        st.stop()

    if results.empty:
        st.info("No recipes matched your filters. Try relaxing the constraints.")
    else:
        if uploaded_file is not None:
            st.success(f"Found **{len(results)}** recipes for your image.")
        else:
            st.success(f"Found **{len(results)}** recipes for: *{query}*")

        has_proj = "x_proj" in results.columns
        if has_proj:
            st.caption("📍 PCA coordinates are attached — check the Embedding Map page to see where these land!")

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            render_recipe_card(row.to_dict(), rank=rank)
