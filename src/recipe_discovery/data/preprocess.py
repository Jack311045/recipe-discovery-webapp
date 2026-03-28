"""Preprocess raw recipe and interaction data."""

from __future__ import annotations

import pandas as pd


def preprocess_tables(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Return a minimally cleaned merged table."""
    if recipes.empty:
        return recipes.copy()

    df = recipes.copy()
    df["ingredient_count"] = 0
    df["step_count"] = 0
    df["avg_rating"] = None

    if not interactions.empty and {"recipe_id", "rating"}.issubset(interactions.columns):
        ratings = (
            interactions.groupby("recipe_id", as_index=False)["rating"]
            .mean()
            .rename(columns={"recipe_id": "id", "rating": "avg_rating"})
        )
        df = df.merge(ratings, on="id", how="left", suffixes=("", "_from_interactions"))
        if "avg_rating_from_interactions" in df.columns:
            df["avg_rating"] = df["avg_rating_from_interactions"]
            df = df.drop(columns=["avg_rating_from_interactions"])

    return df
