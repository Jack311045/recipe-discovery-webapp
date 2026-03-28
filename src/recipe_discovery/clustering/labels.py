"""Helpers for naming or summarizing clusters."""

from __future__ import annotations

import pandas as pd


def summarize_cluster(df: pd.DataFrame, cluster_id: int) -> dict[str, object]:
    """Return a lightweight cluster summary."""
    subset = df[df["cluster"] == cluster_id].copy()
    return {
        "cluster_id": cluster_id,
        "size": len(subset),
    }
