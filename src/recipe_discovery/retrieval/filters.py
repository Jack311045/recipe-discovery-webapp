"""Post-retrieval filtering logic."""

from __future__ import annotations

import re

import pandas as pd

from recipe_discovery.data.schema import get_one_hot_tag_columns

_TIME_BUCKETS: list[tuple[str, int]] = [
    ("15-minutes-or-less", 15),
    ("30-minutes-or-less", 30),
    ("60-minutes-or-less", 60),
    ("4-hours-or-less", 240),
]


def _normalize_token(value: str) -> str:
    """Normalize user and column labels for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower().strip()).strip("-")


def _resolve_dietary_columns(df: pd.DataFrame, dietary_filter: str) -> list[str]:
    """Resolve user dietary terms to one-hot tag column names.

    Supports comma/pipe/slash-separated filters such as
    ``"vegetarian, gluten-free"``.
    """
    tag_columns = get_one_hot_tag_columns(df)
    if not tag_columns:
        return []

    normalized_to_col = {_normalize_token(col): col for col in tag_columns}
    requested = [part.strip() for part in re.split(r"[,|/]", dietary_filter) if part.strip()]
    if not requested:
        return []

    resolved: list[str] = []
    for token in requested:
        norm = _normalize_token(token)
        if norm in normalized_to_col:
            resolved.append(normalized_to_col[norm])
            continue

        partial_matches = [
            col
            for normalized, col in normalized_to_col.items()
            if norm and (norm in normalized or normalized in norm)
        ]
        if len(partial_matches) == 1:
            resolved.append(partial_matches[0])

    # Preserve order but drop duplicates.
    return list(dict.fromkeys(resolved))


def _apply_time_filter(df: pd.DataFrame, max_time_minutes: int) -> pd.DataFrame:
    """Filter by total prep time using minutes first, then one-hot buckets."""
    if "minutes" in df.columns:
        minutes = pd.to_numeric(df["minutes"], errors="coerce")
        return df.loc[minutes <= max_time_minutes]

    masks = []
    for bucket_col, bucket_limit in _TIME_BUCKETS:
        if bucket_col in df.columns and max_time_minutes >= bucket_limit:
            masks.append(df[bucket_col] == 1)

    if not masks:
        return df

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask | mask
    return df.loc[combined_mask]


def _apply_ingredient_filter(df: pd.DataFrame, max_ingredients: int) -> pd.DataFrame:
    """Filter by ingredient count using numeric field first and one-hot fallback."""
    if "n_ingredients" in df.columns:
        ingredient_counts = pd.to_numeric(df["n_ingredients"], errors="coerce")
        return df.loc[ingredient_counts <= max_ingredients]

    if max_ingredients <= 5 and "5-ingredients-or-less" in df.columns:
        return df.loc[df["5-ingredients-or-less"] == 1]
    return df

def _apply_calories_filter(df: pd.DataFrame, max_calories: int) -> pd.DataFrame:
    """Filter by total fat using numeric field."""
    if "calories" in df.columns:
        calories_values = pd.to_numeric(df["calories"], errors="coerce")
        return df.loc[calories_values <= max_calories]
    return df

def _apply_fat_filter(df: pd.DataFrame, max_fat: int) -> pd.DataFrame:
    """Filter by total fat using numeric field."""
    if "total_fat" in df.columns:
        fat_values = pd.to_numeric(df["total_fat"], errors="coerce")
        return df.loc[fat_values <= max_fat]
    return df

def _apply_sugar_filter(df: pd.DataFrame, max_sugar: int) -> pd.DataFrame:
    """Filter by total sugar using numeric field."""
    if "sugar" in df.columns:
        sugar_values = pd.to_numeric(df["sugar"], errors="coerce")
        return df.loc[sugar_values <= max_sugar]
    return df

def _apply_sodium_filter(df: pd.DataFrame, max_sodium: int) -> pd.DataFrame:
    """Filter by total sodium using numeric field."""
    if "sodium" in df.columns:
        sodium_values = pd.to_numeric(df["sodium"], errors="coerce")
        return df.loc[sodium_values <= max_sodium]
    return df

def _apply_protein_filter(df: pd.DataFrame, max_protein: int) -> pd.DataFrame:
    """Filter by total protein using numeric field."""
    if "protein" in df.columns:
        protein_values = pd.to_numeric(df["protein"], errors="coerce")
        return df.loc[protein_values <= max_protein]
    return df

def _apply_saturated_fat_filter(df: pd.DataFrame, max_saturated_fat: int) -> pd.DataFrame:
    """Filter by total saturated fat using numeric field."""
    if "saturated_fat" in df.columns:
        saturated_fat_values = pd.to_numeric(df["saturated_fat"], errors="coerce")
        return df.loc[saturated_fat_values <= max_saturated_fat]
    return df

def _apply_carbohydrates_filter(df: pd.DataFrame, max_carbohydrates: int) -> pd.DataFrame:
    """Filter by total carbohydrates using numeric field."""
    if "carbohydrates" in df.columns:
        carbohydrates_values = pd.to_numeric(df["carbohydrates"], errors="coerce")
        return df.loc[carbohydrates_values <= max_carbohydrates]
    return df

def _apply_rating_filter(df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    """Filter by average rating using numeric field."""
    if "rating" in df.columns:
        rating_values = pd.to_numeric(df["rating"], errors="coerce")
        return df.loc[rating_values >= min_rating]
    return df



def apply_basic_filters(
    df: pd.DataFrame,
    dietary_filter: str | None = None,
    max_time_minutes: int | None = None,
    max_ingredients: int | None = None,
    max_calories: int | None = None,
    max_fat: int | None = None,
    max_sugar: int | None = None,
    max_sodium: int | None = None,
    max_protein: int | None = None,
    max_saturated_fat: int | None = None,
    max_carbohydrates: int | None = None,
    min_rating: float | None = None,
) -> pd.DataFrame:
    """Apply metadata and one-hot dietary filters to a candidate result table."""
    result = df.copy()

    if max_time_minutes is not None:
        result = _apply_time_filter(result, max_time_minutes)

    if max_ingredients is not None:
        result = _apply_ingredient_filter(result, max_ingredients)

    if max_calories is not None:
        result = _apply_calories_filter(result, max_calories)

    if max_fat is not None:
        result = _apply_fat_filter(result, max_fat)

    if max_sugar is not None:
        result = _apply_sugar_filter(result, max_sugar)

    if max_sodium is not None:
        result = _apply_sodium_filter(result, max_sodium)

    if max_protein is not None:
        result = _apply_protein_filter(result, max_protein)

    if max_saturated_fat is not None:
        result = _apply_saturated_fat_filter(result, max_saturated_fat)

    if max_carbohydrates is not None:
        result = _apply_carbohydrates_filter(result, max_carbohydrates)

    if min_rating is not None:
        result = _apply_rating_filter(result, min_rating)

    if dietary_filter and dietary_filter.lower() != "any":
        dietary_cols = _resolve_dietary_columns(result, dietary_filter)
        if dietary_cols:
            mask = result[dietary_cols].fillna(0).astype(int).eq(1).all(axis=1)
            result = result.loc[mask]

    return result.reset_index(drop=True)
