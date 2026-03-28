"""Candidate ranking helpers."""

from __future__ import annotations

import pandas as pd


def attach_scores(df: pd.DataFrame, scores: list[float]) -> pd.DataFrame:
    """Attach similarity scores to a result table."""
    result = df.copy()
    result["similarity_score"] = scores
    return result.sort_values("similarity_score", ascending=False).reset_index(drop=True)
