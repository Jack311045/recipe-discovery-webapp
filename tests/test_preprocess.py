"""Tests for preprocessing."""

from __future__ import annotations

import pandas as pd

from recipe_discovery.data.preprocess import preprocess_tables


def test_preprocess_returns_dataframe() -> None:
    recipes = pd.DataFrame(
        {
            "id": [1],
            "name": ["Soup"],
            "minutes": [30],
            "tags": ["['easy']"],
            "nutrition": ["[100, 1, 2, 3, 4, 5, 6]"],
            "steps": ["['mix', 'cook']"],
            "ingredients": ["['water', 'salt']"],
            "description": ["Simple soup"],
        }
    )
    interactions = pd.DataFrame(columns=["user_id", "recipe_id", "date", "rating", "review"])

    out = preprocess_tables(recipes=recipes, interactions=interactions)
    assert isinstance(out, pd.DataFrame)
    assert "ingredient_count" in out.columns
