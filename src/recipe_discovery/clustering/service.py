"""Cluster browsing utilities."""

from __future__ import annotations

import pandas as pd


def attach_cluster_assignments(df: pd.DataFrame, labels: list[int]) -> pd.DataFrame:
    """Attach cluster labels to metadata."""
    result = df.copy()
    result["cluster"] = labels
    return result
