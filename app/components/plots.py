"""Shared plotting helpers for Streamlit pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    hover_name: str | None = None,
    title: str = "",
    x_label: str = "Dimension 1",
    y_label: str = "Dimension 2",
) -> go.Figure:
    """Return an interactive 2D scatter plot of recipe projections."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_name=hover_name,
        title=title,
        labels={x: x_label, y: y_label},
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        legend_title_text="Group",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def scatter_2d_with_highlights(
    background_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    x: str = "x_proj",
    y: str = "y_proj",
    hover_name: str = "name",
    x_label: str = "PC1",
    y_label: str = "PC2",
    title: str = "Recipe Embedding Map",
) -> go.Figure:
    """Scatter plot with all recipes as grey background + search results highlighted in orange."""
    fig = go.Figure()

    # Background: all recipes (grey)
    fig.add_trace(
        go.Scatter(
            x=background_df[x],
            y=background_df[y],
            mode="markers",
            name="All recipes",
            text=background_df[hover_name] if hover_name in background_df.columns else None,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(color="lightgrey", size=5, opacity=0.5),
        )
    )

    # Foreground: search result highlights
    if not highlight_df.empty:
        fig.add_trace(
            go.Scatter(
                x=highlight_df[x],
                y=highlight_df[y],
                mode="markers+text",
                name="Search results",
                text=highlight_df[hover_name] if hover_name in highlight_df.columns else None,
                textposition="top center",
                hovertemplate="%{text}<extra></extra>",
                marker=dict(color="orangered", size=10, symbol="star"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
