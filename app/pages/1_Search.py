"""Semantic search page."""

from __future__ import annotations

import streamlit as st

st.title("Search Recipes")
st.caption("Semantic retrieval page placeholder.")

query = st.text_input("Describe what you want to eat", placeholder="quick spicy tofu dinner")
diet = st.selectbox("Dietary preference", ["Any", "Vegetarian", "Vegan", "Gluten-free"])
max_time = st.slider("Maximum cooking time (minutes)", min_value=5, max_value=180, value=45)

if st.button("Search"):
    st.success("Search stub triggered.")
    st.write(
        {
            "query": query,
            "dietary_preference": diet,
            "max_time_minutes": max_time,
        }
    )
