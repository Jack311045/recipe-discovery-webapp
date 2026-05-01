"""TF-IDF lexical retrieval artifacts and scoring helpers."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recipe_discovery.settings import ARTIFACTS_DIR
from recipe_discovery.utils.io import ensure_parent_dir

logger = logging.getLogger(__name__)

LEXICAL_INDEX_VERSION = 1
DEFAULT_LEXICAL_INDEX_PATH = ARTIFACTS_DIR / "lexical_index.joblib"


@dataclass(frozen=True)
class LexicalIndex:
    """Persisted TF-IDF vectorizer, sparse matrix, and row-aligned recipe IDs."""

    vectorizer: TfidfVectorizer
    matrix: object
    recipe_ids: pd.Series
    metadata: dict[str, Any]


def normalize_recipe_ids(values: Sequence[object] | pd.Series) -> pd.Series:
    """Normalize recipe IDs to robust string keys for artifact alignment."""
    text = pd.Series(values, dtype=object).astype(str).str.strip()
    return text.str.replace(r"\.0+$", "", regex=True)


def build_lexical_index(
    texts: Sequence[str],
    recipe_ids: Sequence[object] | pd.Series,
    *,
    max_features: int | None = 200_000,
    min_df: int = 1,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> LexicalIndex:
    """Build a TF-IDF lexical index over canonical recipe texts."""
    ids = pd.Series(recipe_ids).reset_index(drop=True)
    text_series = pd.Series(texts, dtype=object).fillna("").astype(str).reset_index(drop=True)
    if len(text_series) != len(ids):
        raise ValueError(
            "Recipe text and ID counts must match: "
            f"texts={len(text_series)}, recipe_ids={len(ids)}"
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(text_series.tolist())
    metadata = {
        "version": LEXICAL_INDEX_VERSION,
        "num_recipes": int(matrix.shape[0]),
        "num_features": int(matrix.shape[1]),
        "max_features": max_features,
        "min_df": min_df,
        "ngram_range": list(ngram_range),
        "stop_words": stop_words,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    logger.info(
        "Built lexical TF-IDF index: rows=%d, features=%d.",
        matrix.shape[0],
        matrix.shape[1],
    )
    return LexicalIndex(
        vectorizer=vectorizer,
        matrix=matrix,
        recipe_ids=ids,
        metadata=metadata,
    )


def save_lexical_index(index: LexicalIndex, path: Path | None = None) -> Path:
    """Save a lexical index artifact."""
    path = Path(path) if path else DEFAULT_LEXICAL_INDEX_PATH
    ensure_parent_dir(path)
    payload = {
        "version": LEXICAL_INDEX_VERSION,
        "vectorizer": index.vectorizer,
        "matrix": index.matrix,
        "recipe_ids": index.recipe_ids.reset_index(drop=True),
        "metadata": index.metadata,
    }
    joblib.dump(payload, path)
    logger.info("Saved lexical index -> %s", path)
    return path


def load_lexical_index(path: Path | None = None) -> LexicalIndex:
    """Load a lexical index artifact."""
    path = Path(path) if path else DEFAULT_LEXICAL_INDEX_PATH
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("Lexical index artifact must be a dictionary payload.")

    required = {"vectorizer", "matrix", "recipe_ids"}
    missing = required - set(payload)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Lexical index artifact is missing required keys: {joined}")

    metadata = payload.get("metadata") or {}
    version = payload.get("version", metadata.get("version"))
    if version != LEXICAL_INDEX_VERSION:
        raise ValueError(
            f"Unsupported lexical index version: expected={LEXICAL_INDEX_VERSION}, got={version}"
        )

    recipe_ids = pd.Series(payload["recipe_ids"]).reset_index(drop=True)
    matrix = payload["matrix"]
    if matrix.shape[0] != len(recipe_ids):
        raise ValueError(
            "Lexical index row count mismatch: "
            f"matrix={matrix.shape[0]}, recipe_ids={len(recipe_ids)}"
        )

    return LexicalIndex(
        vectorizer=payload["vectorizer"],
        matrix=matrix,
        recipe_ids=recipe_ids,
        metadata=dict(metadata),
    )


def align_lexical_index(index: LexicalIndex, target_recipe_ids: pd.Series) -> LexicalIndex:
    """Reorder a lexical index so rows match target recipe ID order."""
    source_ids = normalize_recipe_ids(index.recipe_ids)
    target_ids = normalize_recipe_ids(target_recipe_ids)

    if source_ids.duplicated().any():
        duplicate_count = int(source_ids.duplicated().sum())
        raise ValueError(f"Lexical index has {duplicate_count} duplicate recipe_id values.")

    id_to_row = pd.Series(np.arange(len(source_ids), dtype=int), index=source_ids)
    missing = target_ids[~target_ids.isin(id_to_row.index)]
    if not missing.empty:
        sample = ", ".join(missing.head(5).tolist())
        raise ValueError(
            "Some metadata recipe_ids were not found in lexical index artifacts. "
            f"Missing examples: {sample}"
        )

    row_idx = id_to_row.loc[target_ids].to_numpy()
    return LexicalIndex(
        vectorizer=index.vectorizer,
        matrix=index.matrix[row_idx],
        recipe_ids=pd.Series(target_recipe_ids).reset_index(drop=True),
        metadata=index.metadata,
    )


def score_lexical_query(index: LexicalIndex, query: str) -> np.ndarray:
    """Return TF-IDF cosine scores between a query and indexed recipe texts."""
    query_vec = index.vectorizer.transform([query or ""])
    scores = index.matrix @ query_vec.T
    return np.asarray(scores.toarray()).reshape(-1)
