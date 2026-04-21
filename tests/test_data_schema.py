"""Validation tests for processed-data schema and corpus serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recipe_discovery.data.corpus import build_corpus
from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.schema import (
    ID_COLUMN,
    SHORT_METADATA_COLUMNS,
    TEXT_COLUMNS,
    get_one_hot_tag_columns,
    is_unnamed_column,
)


@pytest.fixture(scope="module")
def processed_df() -> pd.DataFrame:
    """Load processed data once for schema-level checks."""
    return load_processed_recipes()


def test_processed_csv_contains_expected_core_columns(processed_df: pd.DataFrame) -> None:
    expected = {ID_COLUMN, *TEXT_COLUMNS, *SHORT_METADATA_COLUMNS}
    assert expected.issubset(set(processed_df.columns))



def test_unnamed_columns_are_dropped_safely(tmp_path: Path) -> None:
    csv_path = tmp_path / "processed_with_unnamed.csv"
    pd.DataFrame(
        {
            "recipe_id": [1],
            "name": ["demo"],
            "description": ["desc"],
            "ingredients": ["ing"],
            "steps": ["step"],
            "minutes": [10],
            "n_steps": [1],
            "n_ingredients": [2],
            "Unnamed: 0": [999],
            " ": ["drop-me"],
        }
    ).to_csv(csv_path, index=False)

    df = load_processed_recipes(csv_path)
    assert not any(is_unnamed_column(col) for col in df.columns)



def test_one_hot_tag_columns_detected_correctly(processed_df: pd.DataFrame) -> None:
    one_hot = get_one_hot_tag_columns(processed_df)

    assert one_hot
    assert "vegetarian" in one_hot
    assert "vegan" in one_hot
    assert "name" not in one_hot
    assert "minutes" not in one_hot



def test_canonical_recipe_text_does_not_contain_literal_nan() -> None:
    sample = pd.DataFrame(
        {
            "recipe_id": [101],
            "name": ["Simple Soup"],
            "description": [np.nan],
            "ingredients": ["water, salt"],
            "steps": [np.nan],
            "minutes": [15],
            "n_steps": [np.nan],
            "n_ingredients": [2],
            "vegetarian": [1],
            "vegan": [1],
        }
    )

    out = build_corpus(sample)
    text = out.loc[0, "recipe_text"]
    assert "nan" not in text.lower()



def test_canonical_recipe_text_includes_expected_core_fields() -> None:
    sample = pd.DataFrame(
        {
            "recipe_id": [202],
            "name": ["Roasted Veg"],
            "description": ["Easy tray bake"],
            "ingredients": ["carrot, pepper"],
            "steps": ["chop then roast"],
            "minutes": [25],
            "n_steps": [2],
            "n_ingredients": [2],
            "vegetarian": [1],
        }
    )

    out = build_corpus(sample)
    text = out.loc[0, "recipe_text"]

    assert "Name: Roasted Veg" in text
    assert "Description: Easy tray bake" in text
    assert "Ingredients: carrot, pepper" in text
    assert "Steps: chop then roast" in text
    assert "Time: 25 minutes" in text



def test_canonical_recipe_text_does_not_dump_all_reviews_by_default() -> None:
    reviews = "great " * 800
    sample = pd.DataFrame(
        {
            "recipe_id": [303],
            "name": ["Quick Pasta"],
            "description": ["weekday meal"],
            "ingredients": ["pasta, tomato"],
            "steps": ["boil and mix"],
            "minutes": [20],
            "n_steps": [2],
            "n_ingredients": [2],
            "all_reviews": [reviews],
            "vegetarian": [1],
        }
    )

    out = build_corpus(sample)
    text = out.loc[0, "recipe_text"]

    assert "all_reviews" not in text.lower()
    assert reviews[:120] not in text
