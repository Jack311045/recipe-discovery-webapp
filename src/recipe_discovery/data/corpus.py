"""Build canonical recipe text for embedding.

The processed CSV has no single ``tags`` column. Instead, active tags are
reconstructed from hundreds of binary one-hot indicator columns. This module
owns the deterministic text serialization used by the embedding pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from recipe_discovery.data.schema import (
    ID_COLUMN,
    NUTRITION_COLUMNS,
    SHORT_METADATA_COLUMNS,
    TEXT_COLUMNS,
    get_one_hot_tag_columns,
)

logger = logging.getLogger(__name__)


def _clean(value: object) -> str:
    """Return a cleaned string, or empty string for NaN / None."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _reconstruct_tags(row: pd.Series, tag_columns: List[str]) -> str:
    """Return comma-separated tag names where the row has value 1."""
    active = [col for col in tag_columns if row.get(col) == 1]
    return ", ".join(active)


def _format_nutrition(row: pd.Series) -> str:
    """Return a compact nutrition summary string."""
    parts: list[str] = []
    for col in NUTRITION_COLUMNS:
        val = row.get(col)
        if pd.notna(val):
            parts.append(f"{col}={val}")
    return ", ".join(parts)


def serialize_recipe(
    row: pd.Series,
    tag_columns: List[str],
    *,
    include_nutrition: bool = True,
) -> str:
    """Convert one processed-CSV row into a deterministic text block.

    Format::

        Title: <name>
        Description: <description>
        Ingredients: <ingredients>
        Steps: <steps>
        Time: <minutes> minutes
        Step Count: <n_steps>
        Ingredient Count: <n_ingredients>
        Tags: <comma-separated active one-hot labels>
        Nutrition: calories=…, protein=…, …

    Missing values are omitted rather than printed as ``nan``.
    """
    lines: list[str] = []

    # --- text fields ---
    for col in TEXT_COLUMNS:
        label = col.replace("_", " ").title()
        val = _clean(row.get(col))
        if val:
            lines.append(f"{label}: {val}")

    # --- compact metadata ---
    meta_map = {"minutes": "Time", "n_steps": "Step Count", "n_ingredients": "Ingredient Count"}
    for col in SHORT_METADATA_COLUMNS:
        val = row.get(col)
        if pd.notna(val):
            suffix = " minutes" if col == "minutes" else ""
            lines.append(f"{meta_map[col]}: {int(val)}{suffix}")

    # --- one-hot tags ---
    tags_str = _reconstruct_tags(row, tag_columns)
    if tags_str:
        lines.append(f"Tags: {tags_str}")

    # --- nutrition ---
    if include_nutrition:
        nut = _format_nutrition(row)
        if nut:
            lines.append(f"Nutrition: {nut}")

    return "\n".join(lines)


def build_corpus(
    df: pd.DataFrame,
    *,
    include_nutrition: bool = True,
    tag_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add a ``recipe_text`` column with canonical text for each recipe.

    Parameters
    ----------
    df:
        Processed recipe DataFrame (e.g. ``Processed_data_updated2.csv``).
    include_nutrition:
        Whether to append a nutrition summary line.
    tag_columns:
        Pre-computed list of one-hot tag column names. If *None*, they are
        detected automatically via :func:`get_one_hot_tag_columns`.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an added ``recipe_text`` column.
    """
    data = df.copy()
    if tag_columns is None:
        tag_columns = get_one_hot_tag_columns(data)
        logger.info("Detected %d one-hot tag columns.", len(tag_columns))

    data["recipe_text"] = data.apply(
        lambda row: serialize_recipe(row, tag_columns, include_nutrition=include_nutrition),
        axis=1,
    )
    return data
