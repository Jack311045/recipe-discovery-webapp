"""Build recipe text fields for embedding."""

from __future__ import annotations

import pandas as pd

from recipe_discovery.utils.text import build_recipe_text


def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Add a combined text field used for semantic embedding."""
    data = df.copy()
    data["recipe_text"] = data.apply(
        lambda row: build_recipe_text(
            title=str(row.get("name", "")),
            ingredients=str(row.get("ingredients", "")),
            description=str(row.get("description", "")),
            tags=str(row.get("tags", "")),
        ),
        axis=1,
    )
    return data
