"""Tests for search page UI helper logic."""

from __future__ import annotations

import pandas as pd

from app.components.search_ui import (
    build_result_summary,
    get_active_tags,
    infer_tag_columns,
    sort_results_for_display,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recipe_id": ["1", "2", "3"],
            "name": ["A", "B", "C"],
            "similarity_score": [0.80, 0.95, 0.40],
            "minutes": [45, 15, 30],
            "n_ingredients": [10, 4, 7],
            "rating": [4.2, 4.8, 4.8],
            "calories": [500, 350, 450],
            "vegetarian": [1, 0, 1],
            "italian": [0, 1, 0],
        }
    )


def test_build_result_summary_reports_expected_averages() -> None:
    summary = build_result_summary(_sample_results())

    assert summary["count"] == 3
    assert summary["avg_minutes"] == 30.0
    assert round(float(summary["avg_rating"]), 3) == 4.6
    assert summary["avg_calories"] == 433.3333333333333


def test_build_result_summary_handles_missing_optional_columns() -> None:
    df = pd.DataFrame(
        {
            "recipe_id": ["1"],
            "name": ["A"],
            "minutes": [20],
        }
    )

    summary = build_result_summary(df)

    assert summary["count"] == 1
    assert summary["avg_minutes"] == 20.0
    assert summary["avg_rating"] is None
    assert summary["avg_calories"] is None


def test_sort_results_for_display_best_match_uses_similarity() -> None:
    ranked = sort_results_for_display(_sample_results(), "Best match")

    assert ranked["recipe_id"].tolist() == ["2", "1", "3"]


def test_sort_results_for_display_highest_rating_uses_similarity_tiebreak() -> None:
    ranked = sort_results_for_display(_sample_results(), "Highest rating")

    # recipes 2 and 3 share rating=4.8; 2 has higher similarity score.
    assert ranked["recipe_id"].tolist() == ["2", "3", "1"]


def test_sort_results_for_display_fastest_and_fewest() -> None:
    fastest = sort_results_for_display(_sample_results(), "Fastest")
    fewest = sort_results_for_display(_sample_results(), "Fewest ingredients")

    assert fastest["recipe_id"].tolist() == ["2", "3", "1"]
    assert fewest["recipe_id"].tolist() == ["2", "3", "1"]


def test_sort_results_for_display_missing_column_preserves_order() -> None:
    df = pd.DataFrame(
        {
            "recipe_id": ["1", "2", "3"],
            "name": ["A", "B", "C"],
            "similarity_score": [0.2, 0.3, 0.1],
        }
    )

    ranked = sort_results_for_display(df, "Fastest")

    assert ranked["recipe_id"].tolist() == ["1", "2", "3"]


def test_infer_tag_columns_and_get_active_tags() -> None:
    df = _sample_results()
    tag_columns = infer_tag_columns(df)

    row = df.iloc[0].to_dict()
    active_tags = get_active_tags(row, tag_columns, max_tags=5)

    assert "vegetarian" in tag_columns
    assert "italian" in tag_columns
    assert active_tags == ["vegetarian"]
