"""Validation helpers."""

from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise an error if required columns are missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
