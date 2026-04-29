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


def _build_hover_text(df: pd.DataFrame, hover_name: str) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=str)
    
    text = pd.Series("", index=df.index)
    if hover_name in df.columns:
        text += "<b>" + df[hover_name].astype(str).fillna("Unknown") + "</b><br>"
    if "recipe_id" in df.columns:
        text += "ID: " + df["recipe_id"].astype(str) + "<br>"
        
    meta = []
    if "minutes" in df.columns: meta.append("Time: " + df["minutes"].astype(str) + "m")
    if "n_ingredients" in df.columns: meta.append("Ingr: " + df["n_ingredients"].astype(str))
    if meta:
        meta_str = meta[0]
        for m in meta[1:]:
            meta_str += " | " + m
        text += meta_str + "<br>"
        
    nutri = []
    if "calories" in df.columns: nutri.append("Cal: " + df["calories"].astype(str))
    if "protein" in df.columns: nutri.append("Pro: " + df["protein"].astype(str) + "g")
    if "total fat" in df.columns: nutri.append("Fat: " + df["total fat"].astype(str) + "g")
    if nutri:
        nutri_str = nutri[0]
        for n in nutri[1:]:
            nutri_str += " | " + n
        text += nutri_str

    return text.str.rstrip("<br>")


def scatter_2d_with_highlights(
    background_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    x: str = "x_proj",
    y: str = "y_proj",
    hover_name: str = "name",
    x_label: str = "PC1",
    y_label: str = "PC2",
    title: str = "Recipe Embedding Map",
    background_hover: bool = False,
    color: str | None = None,
) -> go.Figure:
    """Scatter plot with all recipes as background + search results highlighted in orange.
    Supports optional cluster coloring and rich hover tooltips.
    """
    if "plot_hover_text" not in background_df.columns:
        background_df = background_df.copy()
        background_df["plot_hover_text"] = _build_hover_text(background_df, hover_name)
    
    if not highlight_df.empty and "plot_hover_text" not in highlight_df.columns:
        highlight_df = highlight_df.copy()
        highlight_df["plot_hover_text"] = _build_hover_text(highlight_df, hover_name)

    # Background: all recipes
    if color and color in background_df.columns:
        fig = px.scatter(
            background_df,
            x=x,
            y=y,
            color=color,
            title=title,
            labels={x: x_label, y: y_label},
            opacity=0.4,
            category_orders={color: sorted(background_df[color].dropna().unique())},
            hover_name=None,
            render_mode="webgl"
        )
        # Update px generated traces with appropriate marker size and hover text
        for trace in fig.data:
            trace.marker.size = 4
            if background_hover:
                trace.text = background_df[background_df[color] == trace.name]["plot_hover_text"]
                trace.hovertemplate = "%{text}<extra></extra>"
            else:
                trace.hoverinfo = "skip"
                trace.hovertemplate = None
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=background_df[x],
                y=background_df[y],
                mode="markers",
                name="All recipes",
                text=background_df["plot_hover_text"] if background_hover else None,
                hovertemplate="%{text}<extra></extra>" if background_hover else None,
                hoverinfo="text" if background_hover else "skip",
                marker=dict(color="rgba(91, 141, 239, 0.28)", size=3),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
        )

    # Foreground: search result highlights
    if not highlight_df.empty:
        fig.add_trace(
            go.Scatter(
                x=highlight_df[x],
                y=highlight_df[y],
                mode="markers",
                name="Search results",
                text=highlight_df["plot_hover_text"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(color="#FF6B00", size=14, symbol="star", line=dict(color="#222222", width=1)),
            )
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02),
        height=700,
    )
    return fig
