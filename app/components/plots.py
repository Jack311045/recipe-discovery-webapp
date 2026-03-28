"""Shared plotting helpers for Streamlit pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def scatter_2d(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    """Return a basic 2D scatter plot."""
    return px.scatter(df, x=x, y=y, color=color)
