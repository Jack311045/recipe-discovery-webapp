"""Column contracts used by the pipeline."""

from __future__ import annotations

import re
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Raw-table column contracts (existing)
# ---------------------------------------------------------------------------

RECIPE_REQUIRED_COLUMNS = [
    "id",
    "name",
    "minutes",
    "tags",
    "nutrition",
    "steps",
    "ingredients",
    "description",
]

INTERACTION_REQUIRED_COLUMNS = [
    "user_id",
    "recipe_id",
    "date",
    "rating",
    "review",
]

# ---------------------------------------------------------------------------
# Processed-CSV column groups (used by the embeddings pipeline)
# ---------------------------------------------------------------------------

ID_COLUMN = "recipe_id"

TEXT_COLUMNS: List[str] = ["name", "description", "ingredients", "steps"]

SHORT_METADATA_COLUMNS: List[str] = ["minutes", "n_steps", "n_ingredients"]

NUTRITION_COLUMNS: List[str] = [
    "calories",
    "total fat",
    "sugar",
    "sodium",
    "protein",
    "saturated fat",
    "carbohydrates",
]

OUTCOME_COLUMNS: List[str] = ["rating", "num_ratings"]

LONG_TEXT_COLUMNS: List[str] = ["all_reviews"]

# Every "known" column that is NOT a one-hot tag indicator
_KNOWN_COLUMNS: set[str] = (
    {ID_COLUMN}
    | set(TEXT_COLUMNS)
    | set(SHORT_METADATA_COLUMNS)
    | set(NUTRITION_COLUMNS)
    | set(OUTCOME_COLUMNS)
    | set(LONG_TEXT_COLUMNS)
)

_UNNAMED_RE = re.compile(r"^Unnamed", re.IGNORECASE)


def is_unnamed_column(col_name: str) -> bool:
    """Return True for unnamed / empty columns produced by pandas CSV I/O."""
    return col_name.strip() == "" or bool(_UNNAMED_RE.match(col_name))


def get_one_hot_tag_columns(df: pd.DataFrame) -> List[str]:
    """Identify binary one-hot tag columns in a processed recipe DataFrame.

    A column is treated as a one-hot tag if it is:
    * not in the set of known direct / metadata / nutrition / outcome columns,
    * not unnamed,
    * and all of its non-null values are in {0, 1}.
    """
    tag_cols: List[str] = []
    for col in df.columns:
        if col in _KNOWN_COLUMNS or is_unnamed_column(col):
            continue
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            continue
        if set(vals).issubset({0, 1, 0.0, 1.0, True, False}):
            tag_cols.append(col)
    return tag_cols
