"""Post-retrieval filtering logic."""

from __future__ import annotations

import pandas as pd


def apply_basic_filters(
    df: pd.DataFrame,
    dietary_filter: str | None = None,
    max_time_minutes: int | None = None,
) -> pd.DataFrame:
    """Apply simple metadata-based filters."""
    result = df.copy()

    if max_time_minutes is not None and "minutes" in result.columns:
        result = result[result["minutes"] <= max_time_minutes]

    if dietary_filter and dietary_filter.lower() != "any" and "tags" in result.columns:
        mask = result["tags"].astype(str).str.contains(dietary_filter, case=False, na=False)
        result = result[mask]

    return result.reset_index(drop=True)
