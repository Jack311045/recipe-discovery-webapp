"""Load raw and processed Food.com tables."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from recipe_discovery.data.schema import (
    INTERACTION_REQUIRED_COLUMNS,
    RECIPE_REQUIRED_COLUMNS,
    is_unnamed_column,
)
from recipe_discovery.settings import DATA_PROCESSED_DIR
from recipe_discovery.utils.validation import require_columns

logger = logging.getLogger(__name__)


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


def load_processed_recipes(path: Path | None = None) -> pd.DataFrame:
    """Load the processed recipe CSV and drop unnamed columns.

    Parameters
    ----------
    path:
        Optional override.  Defaults to
        ``data/processed/Processed_data_updated2.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with unnamed columns removed.
    """
    if path is None:
        path = DATA_PROCESSED_DIR / "Processed_data_updated2.csv"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")

    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if is_unnamed_column(c)]
    if unnamed:
        logger.info("Dropping %d unnamed columns: %s", len(unnamed), unnamed)
        df = df.drop(columns=unnamed)

    logger.info("Loaded processed recipes: %d rows, %d columns.", len(df), len(df.columns))
    return df
