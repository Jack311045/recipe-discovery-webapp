"""Semantic search page."""

from __future__ import annotations

import streamlit as st

st.title("Search Recipes")
st.caption("Semantic retrieval page placeholder.")

query = st.text_input("Describe what you want to eat", placeholder="quick spicy tofu dinner")
diet = st.selectbox("Dietary preference", ["Any", "Vegetarian", "Vegan", "Gluten-free"])
max_time = st.slider("Maximum cooking time (minutes)", min_value=5, max_value=180, value=45)
calories = st.slider("Max calories", 100, 10000, 500)
fat = st.slider("Max total fat (g)", 0, 17183, 20)
sugar = st.slider("Max sugar (g)", 0, 362729, 20)
sodium = st.slider("Max sodium (mg)", 0, 29338, 500)
protein = st.slider("Max protein (g)", 0, 6552, 20)
saturated_fat = st.slider("Max saturated fat (g)", 0, 10395, 10)
carbonhydrates = st.slider("Max carbohydrates (g)", 0, 30698, 50)
total_ratings = st.slider("Average Rating", 0, 5, 4)

if st.button("Search"):
    st.success("Search stub triggered.")
    st.write(
        {
            "query": query,
            "dietary_preference": diet,
            "max_time_minutes": max_time,
            "calories": calories,
            "fat": fat,
            "sugar": sugar,
            "sodium": sodium,
            "protein": protein,
            "saturated_fat": saturated_fat,
            "carbonhydrates": carbonhydrates,
            "total_ratings": total_ratings

        }
    )
