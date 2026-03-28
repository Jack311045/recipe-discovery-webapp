"""Load raw Food.com tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recipe_discovery.data.schema import (
    INTERACTION_REQUIRED_COLUMNS,
    RECIPE_REQUIRED_COLUMNS,
)
from recipe_discovery.utils.validation import require_columns


def _default_paths() -> tuple[Path, Path]:
    recipes_path = Path("data/raw/RAW_recipes.csv")
    interactions_path = Path("data/raw/RAW_interactions.csv")
    return recipes_path, interactions_path


def load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw recipes and interactions tables."""
    recipes_path, interactions_path = _default_paths()

    if not recipes_path.exists():
        recipes = pd.DataFrame(columns=RECIPE_REQUIRED_COLUMNS)
    else:
        recipes = pd.read_csv(recipes_path)

    if not interactions_path.exists():
        interactions = pd.DataFrame(columns=INTERACTION_REQUIRED_COLUMNS)
    else:
        interactions = pd.read_csv(interactions_path)

    require_columns(recipes, RECIPE_REQUIRED_COLUMNS)
    require_columns(interactions, INTERACTION_REQUIRED_COLUMNS)
    return recipes, interactions
