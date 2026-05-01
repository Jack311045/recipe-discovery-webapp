"""UI-oriented helpers for search result presentation and interaction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from recipe_discovery.data.schema import get_one_hot_tag_columns

SORT_OPTIONS = [
    "Best match",
    "Highest rating",
    "Fastest",
    "Fewest ingredients",
]

DEFAULT_SEARCH_RESULT_LIMIT = 14
LOAD_MORE_RESULT_INCREMENT = 6
SEARCH_PREFETCH_RESULT_LIMIT = (
    DEFAULT_SEARCH_RESULT_LIMIT + LOAD_MORE_RESULT_INCREMENT * 2
)


def _as_float(value: object) -> float | None:
    """Convert a value to float, returning None for invalid values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_for_column(df: pd.DataFrame, column: str) -> float | None:
    """Return numeric column mean or None when not available."""
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.dropna().empty:
        return None
    return float(series.mean())


def build_result_summary(results_df: pd.DataFrame) -> dict[str, float | int | None]:
    """Build summary metrics shown above result cards."""
    return {
        "count": int(len(results_df)),
        "avg_minutes": _mean_for_column(results_df, "minutes"),
        "avg_rating": _mean_for_column(results_df, "rating"),
        "avg_calories": _mean_for_column(results_df, "calories"),
    }


def sort_results_for_display(results_df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    """Sort already-returned results for display only.

    This does not trigger retrieval or alter backend ranking logic.
    """
    if results_df.empty:
        return results_df.copy()

    ranked = results_df.copy()

    if sort_mode == "Best match":
        if "similarity_score" in ranked.columns:
            ranked = ranked.sort_values(
                by="similarity_score",
                ascending=False,
                kind="mergesort",
            )
        return ranked.reset_index(drop=True)

    primary_col = {
        "Highest rating": "rating",
        "Fastest": "minutes",
        "Fewest ingredients": "n_ingredients",
    }.get(sort_mode)

    if primary_col is None or primary_col not in ranked.columns:
        return ranked.reset_index(drop=True)

    ranked["_sort_primary"] = pd.to_numeric(ranked[primary_col], errors="coerce")

    ascending = sort_mode in {"Fastest", "Fewest ingredients"}
    sort_cols = ["_sort_primary"]
    sort_ascending = [ascending]

    if "similarity_score" in ranked.columns:
        sort_cols.append("similarity_score")
        sort_ascending.append(False)

    ranked = ranked.sort_values(
        by=sort_cols,
        ascending=sort_ascending,
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_sort_primary"])

    return ranked.reset_index(drop=True)


def merge_appended_results(
    existing_results: pd.DataFrame,
    expanded_results: pd.DataFrame,
    *,
    limit: int,
) -> pd.DataFrame:
    """Append newly fetched results after the recipes already visible.

    Retrieval may slightly reorder rows when the backend is asked for a larger
    pool. The UI load-more action should still feel like appending, so existing
    rows keep their order and only unseen recipes are added below them.
    """
    if existing_results.empty:
        return expanded_results.head(limit).reset_index(drop=True).copy()
    if expanded_results.empty:
        return existing_results.head(limit).reset_index(drop=True).copy()

    key_column = next(
        (
            column
            for column in ("recipe_id", "id", "name")
            if column in existing_results.columns and column in expanded_results.columns
        ),
        None,
    )
    if key_column is None:
        combined = pd.concat([existing_results, expanded_results], ignore_index=True)
        return combined.drop_duplicates().head(limit).reset_index(drop=True)

    visible = existing_results.drop_duplicates(subset=[key_column], keep="first")
    visible = visible.head(limit)
    seen_keys = set(visible[key_column].astype(str))
    new_rows = expanded_results[
        ~expanded_results[key_column].astype(str).isin(seen_keys)
    ]
    combined = pd.concat([visible, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=[key_column], keep="first")
    return combined.head(limit).reset_index(drop=True)


def infer_tag_columns(results_df: pd.DataFrame) -> list[str]:
    """Infer one-hot tag columns from the result frame."""
    if results_df.empty:
        return []
    return get_one_hot_tag_columns(results_df)


def get_active_tags(
    recipe: Mapping[str, object],
    tag_columns: Sequence[str],
    *,
    max_tags: int = 8,
) -> list[str]:
    """Return active one-hot tags for a recipe row."""
    active_tags: list[str] = []
    for col in tag_columns:
        value = _as_float(recipe.get(col))
        if value is not None and int(value) == 1:
            active_tags.append(col)
            if len(active_tags) >= max_tags:
                break
    return active_tags
