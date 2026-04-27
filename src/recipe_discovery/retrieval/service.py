"""End-to-end retrieval service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.schema import ID_COLUMN
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder
from recipe_discovery.embeddings.store import load_embeddings, load_recipe_ids
from recipe_discovery.models.regression import RecipeRegressor
from recipe_discovery.retrieval.filters import apply_basic_filters
from recipe_discovery.retrieval.ranker import compute_combined_ranking
from recipe_discovery.retrieval.similarity import cosine_similarity
from recipe_discovery.settings import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class RetrievalRequest:
    """Search request payload."""

    query: str
    top_k: int = 10
    dietary_filter: str | None = None
    max_time_minutes: int | None = None
    max_ingredients: int | None = None


class RetrievalService:
    """Semantic recipe search service.

    Official v1 runtime behavior uses direct cosine scoring over the in-memory
    embedding matrix. The persisted sklearn index artifact is currently treated
    as a validated optional optimization path, not the default online path.
    """

    def __init__(self) -> None:
        self.encoder = None
        self.embeddings: np.ndarray | None = None
        self.metadata: pd.DataFrame | None = None

    @staticmethod
    def _normalize_recipe_ids(values: pd.Series) -> pd.Series:
        """Normalize recipe IDs to robust string keys for joins."""
        text = values.astype(str).str.strip()
        # CSV parsing can represent integer IDs as float-like strings (for example "58.0").
        return text.str.replace(r"\.0+$", "", regex=True)

    def _get_encoder_config(self) -> EmbeddingConfig:
        """Build encoder config, preferring saved embedding metadata when present."""
        config = EmbeddingConfig()
        metadata_path = ARTIFACTS_DIR / "embedding_metadata.json"
        if not metadata_path.exists():
            return config

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Unable to read %s; falling back to default encoder config.",
                metadata_path,
            )
            return config

        model_name = payload.get("model_name")
        if isinstance(model_name, str) and model_name:
            config.model_name = model_name
        normalize = payload.get("normalize")
        if isinstance(normalize, bool):
            config.normalize = normalize
        return config

    def _align_metadata_by_recipe_id(
        self,
        metadata: pd.DataFrame,
        recipe_ids: pd.Series,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """Align processed metadata rows to embedding row order via recipe_id."""
        if ID_COLUMN not in metadata.columns:
            raise ValueError(f"Processed metadata is missing required column '{ID_COLUMN}'.")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}.")

        if len(recipe_ids) != embeddings.shape[0]:
            raise ValueError(
                "Embeddings and recipe_ids row count mismatch: "
                f"embeddings={embeddings.shape[0]}, recipe_ids={len(recipe_ids)}"
            )

        metadata_ids = self._normalize_recipe_ids(metadata[ID_COLUMN])
        ids = self._normalize_recipe_ids(recipe_ids)

        if metadata_ids.duplicated().any():
            duplicate_count = int(metadata_ids.duplicated().sum())
            raise ValueError(
                f"Processed metadata has {duplicate_count} duplicate recipe_id values."
            )

        if ids.duplicated().any():
            duplicate_count = int(ids.duplicated().sum())
            raise ValueError(
                f"recipe_ids artifact has {duplicate_count} duplicate recipe_id values."
            )

        metadata_normalized = metadata.copy()
        metadata_normalized[ID_COLUMN] = metadata_ids
        id_frame = pd.DataFrame({ID_COLUMN: ids, "_embedding_row": np.arange(len(ids), dtype=int)})
        aligned = id_frame.merge(
            metadata_normalized,
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        missing_ids = aligned.loc[aligned["_merge"] == "left_only", ID_COLUMN]
        if not missing_ids.empty:
            sample = ", ".join(missing_ids.head(5).tolist())
            raise ValueError(
                "Some recipe_ids from artifacts were not found in processed metadata. "
                f"Missing examples: {sample}"
            )

        aligned = (
            aligned.sort_values("_embedding_row")
            .drop(columns=["_embedding_row", "_merge"])
            .reset_index(drop=True)
        )
        return aligned

    def load(
        self,
        *,
        processed_path: Path | None = None,
        embeddings_path: Path | None = None,
        recipe_ids_path: Path | None = None,
    ) -> None:
        """Load encoder, embeddings, recipe_ids, and recipe-aligned metadata.

        Parameters
        ----------
        processed_path:
            Optional override for processed CSV path.
        embeddings_path:
            Optional override for embedding ``.npy`` path.
        recipe_ids_path:
            Optional override for row-aligned recipe IDs path.
        """
        metadata = load_processed_recipes(processed_path)
        embeddings = load_embeddings(embeddings_path)
        recipe_ids = load_recipe_ids(recipe_ids_path)

        self.metadata = self._align_metadata_by_recipe_id(metadata, recipe_ids, embeddings)
        self.embeddings = embeddings

        encoder_config = self._get_encoder_config()
        self.encoder = RecipeEncoder(config=encoder_config)
        self.encoder.load()

        logger.info(
            "RetrievalService loaded: %d rows aligned to %d embedding vectors.",
            len(self.metadata),
            self.embeddings.shape[0],
        )

        # Attach 2D projections if available
        projections_path = ARTIFACTS_DIR / "projections_2d.npy"
        pca_projections_path = ARTIFACTS_DIR / "pca_projection.npy"
        
        proj_file = projections_path if projections_path.exists() else pca_projections_path
        if proj_file.exists():
            try:
                proj_array = np.load(proj_file)
                if len(proj_array) == len(self.metadata):
                    self.metadata["x_proj"] = proj_array[:, 0]
                    self.metadata["y_proj"] = proj_array[:, 1]
                    logger.info("Loaded 2D projections from %s", proj_file)
                else:
                    logger.warning("Projection shape mismatch. Skipping 2D coords.")
            except Exception as e:
                logger.error("Failed to load projections: %s", e)
        else:
            logger.warning("No 2D projections found in artifacts.")

    def get_all_projections(self) -> pd.DataFrame:
        """Return all available 2D projections for the background scatter plot.
        
        Returns a DataFrame with ['recipe_id', 'x_proj', 'y_proj'].
        Returns an empty DataFrame if projections are not loaded.
        """
        if self.metadata is None or "x_proj" not in self.metadata.columns:
            return pd.DataFrame(columns=[ID_COLUMN, "x_proj", "y_proj"])
            
        # Return only the rows that actually have projection coordinates
        has_proj = self.metadata["x_proj"].notna()
        cols = [ID_COLUMN, "x_proj", "y_proj"]
        
        # We also might want to return title or other metadata for tooltips, but the spec
        # specifically requested the background points. Adding title to be safe for UI.
        if "name" in self.metadata.columns:
            cols.append("name")
            
        return self.metadata.loc[has_proj, cols].copy().reset_index(drop=True)


    def _search_candidates(
        self,
        request: RetrievalRequest,
        *,
        limit_to_top_k: bool,
    ) -> pd.DataFrame:
        """Return filtered candidate matches for a query."""
        if self.encoder is None or self.embeddings is None or self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        if request.top_k <= 0:
            return self.metadata.iloc[0:0].assign(similarity_score=pd.Series(dtype=float))

        encoded = self.encoder.encode([request.query], show_progress=False)
        query_vec = np.asarray(encoded)[0]
        scores = cosine_similarity(query=query_vec, matrix=self.embeddings)

        candidate_pool = min(len(scores), max(request.top_k * 10, request.top_k))
        if candidate_pool == 0:
            return self.metadata.iloc[0:0].assign(similarity_score=pd.Series(dtype=float))

        # Candidate pooling prevents filters from exhausting a too-small pre-truncated set.
        top_idx = np.argpartition(-scores, candidate_pool - 1)[:candidate_pool]
        ranked_idx = top_idx[np.argsort(-scores[top_idx])]

        candidates = self.metadata.iloc[ranked_idx].copy()
        candidates["similarity_score"] = scores[ranked_idx]
        filtered = apply_basic_filters(
            candidates,
            dietary_filter=request.dietary_filter,
            max_time_minutes=request.max_time_minutes,
            max_ingredients=request.max_ingredients,
        )

        if filtered.empty:
            return filtered

        ranked = filtered.sort_values("similarity_score", ascending=False)
        if limit_to_top_k:
            ranked = ranked.head(request.top_k)

        return ranked.reset_index(drop=True)

    @staticmethod
    def load_regression_model(
        regression_model_path: Path | None = None,
    ) -> RecipeRegressor | None:
        """Load an optional regression model artifact if present.

        If the artifact path does not exist, ``None`` is returned so callers can
        safely fall back to similarity-only ranking.
        """
        model_path = Path(regression_model_path or (ARTIFACTS_DIR / "regressor.joblib"))
        if not model_path.exists():
            return None
        return RecipeRegressor.load(model_path)

    def rerank_candidates(
        self,
        candidates: pd.DataFrame,
        *,
        regression_model: object | None = None,
        feature_columns: Sequence[str] | None = None,
        similarity_weight: float = 0.8,
        rating_weight: float = 0.2,
        regression_model_path: Path | None = None,
    ) -> pd.DataFrame:
        """Optionally rerank candidate rows using a regression quality signal.

        This method is additive by design. If no regression model is available,
        similarity-first ranking is preserved.
        """
        model = regression_model
        if model is None and regression_model_path is not None:
            model = self.load_regression_model(regression_model_path)

        return compute_combined_ranking(
            candidates,
            regression_model=model,
            feature_columns=feature_columns,
            similarity_weight=similarity_weight,
            rating_weight=rating_weight,
        )

    def search_with_optional_rerank(
        self,
        request: RetrievalRequest,
        *,
        regression_model: object | None = None,
        feature_columns: Sequence[str] | None = None,
        similarity_weight: float = 0.8,
        rating_weight: float = 0.2,
        regression_model_path: Path | None = None,
    ) -> pd.DataFrame:
        """Search then optionally rerank with regression as an additive layer.

        Official default runtime remains similarity-first and is exposed by
        :meth:`search`. This method is an explicit optional integration path.
        """
        candidates = self._search_candidates(request, limit_to_top_k=False)
        if candidates.empty:
            return candidates

        reranked = self.rerank_candidates(
            candidates,
            regression_model=regression_model,
            feature_columns=feature_columns,
            similarity_weight=similarity_weight,
            rating_weight=rating_weight,
            regression_model_path=regression_model_path,
        )
        return reranked.head(request.top_k).reset_index(drop=True)

    def search(self, request: RetrievalRequest) -> pd.DataFrame:
        """Return filtered top-k recipe matches for a query.

        This v1 runtime path intentionally performs direct cosine scoring
        against the loaded embedding matrix before candidate-pool filtering.
        """
        return self._search_candidates(request, limit_to_top_k=True)
