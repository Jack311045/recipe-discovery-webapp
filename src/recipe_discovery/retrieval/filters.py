"""Post-retrieval filtering logic."""

from __future__ import annotations

import pandas as pd


import re

def apply_basic_filters(
    df: pd.DataFrame,
    dietary_filter: str | None = None,
    max_time_minutes: int | None = None,
    max_ingreidents: int | None = None
) -> pd.DataFrame:
    """Apply simple metadata-based filters."""
    result = df.copy()

    if max_time_minutes is not None:
        if max_time_minutes <= 15 and "15-minutes-or-less" in result.columns:
            result = result[
            (result["15-minutes-or-less"] == 1) | 
            (result['minutes']<=15)] 
        if max_time_minutes <= 30 and "30-minutes-or-less" in result.columns:
            result = result[
            (result["15-minutes-or-less"] == 1) | 
            (result["30-minutes-or-less"] == 1) |
            (result['minutes']<=30)] 
        if max_time_minutes <= 60 and "60-minutes-or-less" in result.columns:
            result = result[
            (result["60-minutes-or-less"] == 1) | 
            (result["30-minutes-or-less"] == 1) | 
            (result["15-minutes-or-less"] == 1) | 
            (result["minutes"] <= 60)
            ]
        if max_time_minutes <= 240 and "4-hours-or-less" in result.columns:
            result = result[
            (result["4-hours-or-less"] == 1) | 
            (result["60-minutes-or-less"] == 1) | 
            (result["30-minutes-or-less"] == 1) | 
            (result["15-minutes-or-less"] == 1)|
            (result["minutes"] <= 240)]
    
        if "minutes" in result.columns:
            result = result[result["minutes"] <= max_time_minutes]

    if max_ingreidents is not None:
        if "n_ingredients" in result.columns:
            result = result[result["n_ingredients"] <= max_ingreidents]
        if max_ingreidents < 5 and "5-ingredients-or-less" in result.columns:
            result = result[result["5-ingredients-or-less"] == 1]

        # Normalize user input
        user_input = user_input.lower().replace("min", "minutes").replace("hr", "hours")
 
        
    if dietary_filter and dietary_filter.lower() != "any" and "tags" in result.columns:
        mask = result["tags"].astype(str).str.contains(dietary_filter, case=False, na=False)
        result = result[mask]


    return result.reset_index(drop=True)      

