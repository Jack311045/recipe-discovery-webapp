"""End-to-end retrieval service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.schema import ID_COLUMN
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder
from recipe_discovery.embeddings.store import load_embeddings, load_recipe_ids
from recipe_discovery.models.regression import RecipeRegressor
from recipe_discovery.retrieval.filters import apply_basic_filters
from recipe_discovery.retrieval.ranker import compute_combined_ranking
from recipe_discovery.retrieval.similarity import cosine_similarity
from recipe_discovery.settings import ARTIFACTS_DIR, DATA_PROCESSED_DIR

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
SIGLIP_COMBINED_DEFAULT_IMAGE_WEIGHT = 0.75
SIGLIP_COMBINED_CANDIDATE_MULTIPLIER = 50
FALLBACK_IMAGE_URL = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c"


logger = logging.getLogger(__name__)


@dataclass
class RetrievalRequest:
    """Search request payload."""

    query: str | None = None
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
        self._siglip_embeddings: np.ndarray | None = None
        self._siglip_processor: AutoProcessor | None = None
        self._siglip_model: AutoModel | None = None
        self._siglip_device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def _align_embeddings_to_metadata(
        self,
        embeddings: np.ndarray,
        recipe_ids: pd.Series,
    ) -> np.ndarray:
        """Reorder embeddings so their rows match ``self.metadata`` order."""
        if self.metadata is None:
            raise RuntimeError("Metadata must be loaded before aligning embeddings.")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}.")

        ids = self._normalize_recipe_ids(recipe_ids)
        if ids.duplicated().any():
            duplicate_count = int(ids.duplicated().sum())
            raise ValueError(
                f"recipe_ids artifact has {duplicate_count} duplicate recipe_id values."
            )

        meta_ids = self._normalize_recipe_ids(self.metadata[ID_COLUMN])
        id_to_row = pd.Series(np.arange(len(ids), dtype=int), index=ids)
        missing = meta_ids[~meta_ids.isin(id_to_row.index)]
        if not missing.empty:
            sample = ", ".join(missing.head(5).tolist())
            raise ValueError(
                "Some metadata recipe_ids were not found in SigLIP artifacts. "
                f"Missing examples: {sample}"
            )

        row_idx = id_to_row.loc[meta_ids].to_numpy()
        return embeddings[row_idx]

    def _load_image_map(self, path: Path) -> pd.Series:
        if not path.exists():
            logger.info("Image map not found at %s; using fallback images.", path)
            return pd.Series(dtype=str)

        image_df = pd.read_parquet(path)
        if "recipe_id" not in image_df.columns or "image_url" not in image_df.columns:
            raise ValueError("image_map.parquet must include recipe_id and image_url columns.")

        image_df = image_df[["recipe_id", "image_url"]].copy()
        image_df["recipe_id"] = self._normalize_recipe_ids(image_df["recipe_id"])
        image_df = image_df.drop_duplicates(subset=["recipe_id"], keep="first")
        return image_df.set_index("recipe_id")["image_url"]

    def _attach_image_urls(self) -> None:
        if self.metadata is None:
            return

        image_map_path = DATA_PROCESSED_DIR / "image_map.parquet"
        image_map = self._load_image_map(image_map_path)
        if image_map.empty:
            self.metadata["image_url"] = FALLBACK_IMAGE_URL
            return

        meta_ids = self._normalize_recipe_ids(self.metadata[ID_COLUMN])
        self.metadata["image_url"] = meta_ids.map(image_map).fillna(FALLBACK_IMAGE_URL)

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
        self._attach_image_urls()

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

    def _load_siglip_embeddings(self) -> None:
        if self._siglip_embeddings is not None:
            return

        emb_path = ARTIFACTS_DIR / "recipe_embeddings_siglip.npy"
        ids_path = ARTIFACTS_DIR / "recipe_ids_siglip.csv"
        if not emb_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                "SigLIP artifacts not found. Run scripts/generate_siglip_embeddings.py first."
            )

        siglip_embeddings = load_embeddings(emb_path)
        siglip_ids = load_recipe_ids(ids_path)
        self._siglip_embeddings = self._align_embeddings_to_metadata(siglip_embeddings, siglip_ids)

    def _load_siglip_model(self) -> None:
        if self._siglip_model is not None and self._siglip_processor is not None:
            return

        self._siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
        self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(self._siglip_device)
        self._siglip_model.eval()

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        self._load_siglip_model()
        if self._siglip_model is None or self._siglip_processor is None:
            raise RuntimeError("SigLIP model failed to load.")

        with torch.no_grad():
            inputs = self._siglip_processor(images=image.convert("RGB"), return_tensors="pt").to(
                self._siglip_device
            )
            outputs = self._siglip_model.get_image_features(**inputs)
            features = outputs.pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def _encode_siglip_text(self, text: str) -> np.ndarray:
        self._load_siglip_model()
        if self._siglip_model is None or self._siglip_processor is None:
            raise RuntimeError("SigLIP model failed to load.")

        with torch.no_grad():
            inputs = self._siglip_processor(
                text=[text],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self._siglip_device)
            outputs = self._siglip_model.get_text_features(**inputs)
            features = outputs.pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def encode_combined(
        self,
        text: str,
        image: Image.Image,
        *,
        alpha: float = SIGLIP_COMBINED_DEFAULT_IMAGE_WEIGHT,
    ) -> np.ndarray:
        """Return a combined SigLIP vector for text + image inputs."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        self._load_siglip_model()
        if self._siglip_model is None or self._siglip_processor is None:
            raise RuntimeError("SigLIP model failed to load.")

        with torch.no_grad():
            img_inputs = self._siglip_processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            ).to(self._siglip_device)
            img_out = self._siglip_model.get_image_features(**img_inputs)
            img_vec = img_out.pooler_output
            img_vec = img_vec / img_vec.norm(p=2, dim=-1, keepdim=True)

            txt_inputs = self._siglip_processor(
                text=[text],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self._siglip_device)
            txt_out = self._siglip_model.get_text_features(**txt_inputs)
            txt_vec = txt_out.pooler_output
            txt_vec = txt_vec / txt_vec.norm(p=2, dim=-1, keepdim=True)

        combined = alpha * img_vec + (1.0 - alpha) * txt_vec
        combined = combined / combined.norm(p=2, dim=-1, keepdim=True)
        return combined.cpu().numpy()[0]

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


    def _search_candidates_for_vector(
        self,
        request: RetrievalRequest,
        *,
        query_vec: np.ndarray,
        embeddings: np.ndarray,
        limit_to_top_k: bool,
    ) -> pd.DataFrame:
        """Return filtered candidate matches for a query vector."""
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        if request.top_k <= 0:
            return self.metadata.iloc[0:0].assign(similarity_score=pd.Series(dtype=float))

        scores = cosine_similarity(query=query_vec, matrix=embeddings)

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

    def _search_candidates(
        self,
        request: RetrievalRequest,
        *,
        limit_to_top_k: bool,
    ) -> pd.DataFrame:
        """Return filtered candidate matches for a text query."""
        if self.encoder is None or self.embeddings is None or self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        if not request.query or not request.query.strip():
            raise ValueError("Search query is required for text search.")

        encoded = self.encoder.encode([request.query], show_progress=False)
        query_vec = np.asarray(encoded)[0]
        return self._search_candidates_for_vector(
            request,
            query_vec=query_vec,
            embeddings=self.embeddings,
            limit_to_top_k=limit_to_top_k,
        )

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

    def search_by_image(self, image: Image.Image, request: RetrievalRequest) -> pd.DataFrame:
        """Return filtered top-k recipe matches for an uploaded image."""
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        self._load_siglip_embeddings()
        if self._siglip_embeddings is None:
            raise RuntimeError("SigLIP embeddings failed to load.")

        query_vec = self._encode_image(image)
        return self._search_candidates_for_vector(
            request,
            query_vec=query_vec,
            embeddings=self._siglip_embeddings,
            limit_to_top_k=True,
        )

    def search_combined(
        self,
        text: str,
        image: Image.Image,
        request: RetrievalRequest,
        *,
        alpha: float = SIGLIP_COMBINED_DEFAULT_IMAGE_WEIGHT,
    ) -> pd.DataFrame:
        """Return filtered top-k matches for combined text + image input."""
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")
        if not text.strip():
            raise ValueError("Text query is required for combined search.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        self._load_siglip_embeddings()
        if self._siglip_embeddings is None:
            raise RuntimeError("SigLIP embeddings failed to load.")

        image_vec = self._encode_image(image)
        text_vec = self._encode_siglip_text(text)
        image_scores = cosine_similarity(query=image_vec, matrix=self._siglip_embeddings)
        text_scores = cosine_similarity(query=text_vec, matrix=self._siglip_embeddings)
        combined_scores = alpha * image_scores + (1.0 - alpha) * text_scores

        candidate_pool = min(
            len(image_scores),
            max(request.top_k * SIGLIP_COMBINED_CANDIDATE_MULTIPLIER, request.top_k),
        )
        if request.top_k <= 0 or candidate_pool == 0:
            return self.metadata.iloc[0:0].assign(
                similarity_score=pd.Series(dtype=float),
                image_similarity_score=pd.Series(dtype=float),
                text_similarity_score=pd.Series(dtype=float),
            )

        # Combined queries should treat text as a refinement on visually relevant dishes.
        image_top_idx = np.argpartition(-image_scores, candidate_pool - 1)[:candidate_pool]
        ranked_idx = image_top_idx[np.argsort(-combined_scores[image_top_idx])]

        candidates = self.metadata.iloc[ranked_idx].copy()
        candidates["similarity_score"] = combined_scores[ranked_idx]
        candidates["image_similarity_score"] = image_scores[ranked_idx]
        candidates["text_similarity_score"] = text_scores[ranked_idx]
        filtered = apply_basic_filters(
            candidates,
            dietary_filter=request.dietary_filter,
            max_time_minutes=request.max_time_minutes,
            max_ingredients=request.max_ingredients,
        )

        if filtered.empty:
            return filtered

        ranked = filtered.sort_values("similarity_score", ascending=False)
        return ranked.head(request.top_k).reset_index(drop=True)
