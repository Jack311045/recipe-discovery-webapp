"""Candidate ranking helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def attach_scores(df: pd.DataFrame, scores: list[float]) -> pd.DataFrame:
    """Attach similarity scores to a result table."""
    if len(df) != len(scores):
        raise ValueError(
            "Similarity score count must match candidate rows: "
            f"rows={len(df)}, scores={len(scores)}"
        )

    result = df.copy()
    result["similarity_score"] = scores
    return result.sort_values("similarity_score", ascending=False).reset_index(drop=True)


def _normalize_min_max(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(max_value, min_value):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - min_value) / (max_value - min_value)


def attach_predicted_ratings(
    df: pd.DataFrame,
    model: object | None,
    feature_columns: Sequence[str],
    *,
    output_column: str = "predicted_rating",
) -> pd.DataFrame:
    """Attach predicted rating scores if a regression model is available."""
    result = df.copy()
    if model is None:
        return result

    if not feature_columns:
        raise ValueError("Feature columns are required to attach predicted ratings.")

    missing = [col for col in feature_columns if col not in result.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing feature columns for predicted ratings: {joined}")

    features = result.loc[:, list(feature_columns)].astype(float).to_numpy()
    if not hasattr(model, "predict"):
        raise TypeError("Regression model must define a predict method.")

    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.shape != (len(result),):
        raise ValueError(
            "Predicted rating output shape mismatch: "
            f"expected={(len(result),)}, got={predictions.shape}"
        )

    result[output_column] = predictions
    return result


def compute_combined_ranking(
    df: pd.DataFrame,
    regression_model: object | None,
    feature_columns: Sequence[str] | None,
    *,
    similarity_weight: float = 0.8,
    rating_weight: float = 0.2,
) -> pd.DataFrame:
    """Compute additive ranking score using similarity plus optional predicted rating."""
    if "similarity_score" not in df.columns:
        raise ValueError("Candidate table must include 'similarity_score' column.")
    if similarity_weight < 0 or rating_weight < 0:
        raise ValueError("similarity_weight and rating_weight must be non-negative.")
    if similarity_weight == 0 and rating_weight == 0:
        raise ValueError("At least one ranking weight must be positive.")

    result = df.copy()
    result["normalized_similarity_score"] = _normalize_min_max(result["similarity_score"])

    if regression_model is None:
        # Stable fallback keeps similarity-only behavior unchanged.
        result["combined_score"] = result["normalized_similarity_score"]
        return (
            result.sort_values(
                ["combined_score", "similarity_score"],
                ascending=False,
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    if feature_columns is None:
        raise ValueError("feature_columns is required when regression_model is provided.")

    result = attach_predicted_ratings(result, regression_model, feature_columns)
    result["normalized_predicted_rating"] = _normalize_min_max(result["predicted_rating"])
    result["combined_score"] = (
        similarity_weight * result["normalized_similarity_score"]
        + rating_weight * result["normalized_predicted_rating"]
    )

    return (
        result.sort_values(
            ["combined_score", "similarity_score", "predicted_rating"],
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)
    )
