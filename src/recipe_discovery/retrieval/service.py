"""End-to-end retrieval service."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

try:  # Optional at import time; required only for SigLIP image search.
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None

try:  # Optional at import time; required only for SigLIP image search.
    from transformers import AutoModel, AutoProcessor
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    AutoModel = None
    AutoProcessor = None

from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.schema import ID_COLUMN, get_one_hot_tag_columns
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder
from recipe_discovery.embeddings.store import load_embeddings, load_recipe_ids
from recipe_discovery.models.regression import RecipeRegressor
from recipe_discovery.retrieval.filters import apply_basic_filters
from recipe_discovery.retrieval.image_fetcher import attach_foodcom_images
from recipe_discovery.retrieval.lexical import (
    DEFAULT_LEXICAL_INDEX_PATH,
    LexicalIndex,
    align_lexical_index,
    load_lexical_index,
    score_lexical_query,
)
from recipe_discovery.retrieval.ranker import compute_combined_ranking
from recipe_discovery.retrieval.similarity import cosine_similarity
from recipe_discovery.settings import ARTIFACTS_DIR, DATA_PROCESSED_DIR

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
SIGLIP_COMBINED_DEFAULT_IMAGE_WEIGHT = 0.75
SIGLIP_COMBINED_CANDIDATE_MULTIPLIER = 50
FALLBACK_IMAGE_URL = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c"


logger = logging.getLogger(__name__)

_SMALL_ARTIFACT_WARN_THRESHOLD = 5000
_EXACT_TAG_SCORE_BOOST = 1.0
_DEFAULT_CANDIDATE_POOL_MULTIPLIER = 10
_INTENT_FILTER_CANDIDATE_POOL_MULTIPLIER = 50
_HYBRID_SEMANTIC_WEIGHT = 0.70
_HYBRID_KEYWORD_WEIGHT = 0.30
_NEGATION_TOKENS = {
    "avoid",
    "except",
    "exclude",
    "excluding",
    "minus",
    "no",
    "non",
    "not",
    "without",
}
_NEGATION_SKIP_TOKENS = {"a", "an", "any", "the"}
_MODIFIER_TOKENS = {
    "free",
    "less",
    "light",
    "lite",
    "low",
    "no",
    "non",
    "reduced",
    "without",
}
_MODIFIER_PREFIX_TOKENS = {"light", "lite", "low", "no", "non", "reduced", "without"}
_MODIFIER_SUFFIX_TOKENS = {"free", "less"}
_NUTRITION_TOKEN_ALIASES = {
    "calorie": "calorie",
    "calories": "calorie",
    "carb": "carbohydrate",
    "carbs": "carbohydrate",
    "carbohydrate": "carbohydrate",
    "carbohydrates": "carbohydrate",
    "fat": "fat",
    "protein": "protein",
    "salt": "sodium",
    "sodium": "sodium",
    "sugar": "sugar",
}


@dataclass
class RetrievalRequest:
    """Search request payload."""

    query: str | None = None
    top_k: int = 10
    dietary_filter: str | None = None
    max_time_minutes: int | None = None
    max_ingredients: int | None = None
    max_calories: int | None = None
    max_fat: int | None = None
    max_sugar: int | None = None
    max_sodium: int | None = None
    max_protein: int | None = None
    max_saturated_fat: int | None = None
    max_carbohydrates: int | None = None
    min_rating: float | None = None


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
        self.one_hot_tag_columns: list[str] = []
        self._lexical_index: LexicalIndex | None = None
        self._siglip_embeddings: np.ndarray | None = None
        self._siglip_processor: AutoProcessor | None = None
        self._siglip_model: AutoModel | None = None
        self._siglip_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self._text_query_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _normalize_recipe_ids(values: pd.Series) -> pd.Series:
        """Normalize recipe IDs to robust string keys for joins."""
        text = values.astype(str).str.strip()
        # CSV parsing can represent integer IDs as float-like strings (for example "58.0").
        return text.str.replace(r"\.0+$", "", regex=True)

    @staticmethod
    def _normalize_token(value: str) -> str:
        """Normalize label/query text for robust one-hot tag matching."""
        return re.sub(r"[^a-z0-9]+", "-", value.lower().strip()).strip("-")

    def _normalized_tag_map(self, df: pd.DataFrame) -> dict[str, str]:
        tag_columns = self.one_hot_tag_columns or get_one_hot_tag_columns(df)
        if not tag_columns:
            return {}
        return {self._normalize_token(col): col for col in tag_columns}

    def _resolve_normalized_tag(
        self,
        normalized_query: str,
        normalized_to_col: dict[str, str],
    ) -> str | None:
        if not normalized_query:
            return None
        if normalized_query in normalized_to_col:
            return normalized_to_col[normalized_query]

        # Near-exact tolerance for singular/plural and common query suffixes.
        candidates = {normalized_query}
        if normalized_query.endswith("s") and len(normalized_query) > 1:
            candidates.add(normalized_query[:-1])
        else:
            candidates.add(f"{normalized_query}s")

        for suffix in ("-recipe", "-recipes", "-food", "-foods", "-dish", "-dishes"):
            if normalized_query.endswith(suffix):
                trimmed = normalized_query[: -len(suffix)]
                if trimmed:
                    candidates.add(trimmed)

        for candidate in candidates:
            if candidate in normalized_to_col:
                return normalized_to_col[candidate]
        return None

    def _resolve_query_tag_column(self, query: str, df: pd.DataFrame) -> str | None:
        """Resolve exact/near-exact query intent to a known one-hot tag column."""
        normalized_to_col = self._normalized_tag_map(df)
        return self._resolve_normalized_tag(
            self._normalize_token(query),
            normalized_to_col,
        )

    def _query_tokens(self, query: str | None) -> list[str]:
        normalized = self._normalize_token(query or "")
        return [token for token in normalized.split("-") if token]

    def _resolve_tag_from_tokens(
        self,
        tokens: list[str],
        start: int,
        normalized_to_col: dict[str, str],
    ) -> tuple[str | None, int]:
        """Resolve a tag column from a token span, preferring the longest span."""
        while start < len(tokens) and tokens[start] in _NEGATION_SKIP_TOKENS:
            start += 1
        if start >= len(tokens):
            return None, start

        for end in range(len(tokens), start, -1):
            candidate = "-".join(tokens[start:end])
            tag_col = self._resolve_normalized_tag(candidate, normalized_to_col)
            if tag_col is not None:
                return tag_col, end
        return None, start

    def _resolve_negated_query_tag_columns(
        self,
        query: str | None,
        df: pd.DataFrame,
    ) -> list[str]:
        """Return tag columns that should be excluded by negated query phrasing."""
        tokens = self._query_tokens(query)
        if not tokens:
            return []

        normalized_to_col = self._normalized_tag_map(df)
        if not normalized_to_col:
            return []

        # Exact tag queries such as "no-cook" should stay positive tag intents.
        normalized_query = "-".join(tokens)
        if self._resolve_normalized_tag(normalized_query, normalized_to_col):
            return []

        excluded: list[str] = []
        for idx, token in enumerate(tokens):
            if token not in _NEGATION_TOKENS:
                continue
            tag_col, _ = self._resolve_tag_from_tokens(tokens, idx + 1, normalized_to_col)
            if tag_col is not None:
                excluded.append(tag_col)
        return list(dict.fromkeys(excluded))

    @staticmethod
    def _canonical_nutrition_token(token: str) -> str | None:
        return _NUTRITION_TOKEN_ALIASES.get(token)

    def _resolve_modifier_conflict_columns(
        self,
        query: str | None,
        df: pd.DataFrame,
    ) -> list[str]:
        """Find modifier tags that contradict a bare positive nutrient query.

        Dense embeddings often treat "fat" and "low fat" as close because they
        share the core concept. For bare nutrient searches, keep explicit
        low/no/free variants from dominating unless the user asked for them.
        """
        tokens = self._query_tokens(query)
        if not tokens or any(token in _MODIFIER_TOKENS for token in tokens):
            return []

        normalized_to_col = self._normalized_tag_map(df)
        if not normalized_to_col:
            return []
        normalized_query = "-".join(tokens)
        if self._resolve_normalized_tag(normalized_query, normalized_to_col):
            return []

        query_nutrients = {
            canonical
            for token in tokens
            if (canonical := self._canonical_nutrition_token(token)) is not None
        }
        if not query_nutrients:
            return []

        conflicts: list[str] = []
        for normalized_tag, col in normalized_to_col.items():
            tag_tokens = [token for token in normalized_tag.split("-") if token]
            if not tag_tokens:
                continue
            has_negating_modifier = (
                tag_tokens[0] in _MODIFIER_PREFIX_TOKENS
                or tag_tokens[-1] in _MODIFIER_SUFFIX_TOKENS
            )
            if not has_negating_modifier:
                continue

            tag_nutrients = {
                canonical
                for token in tag_tokens
                if (canonical := self._canonical_nutrition_token(token)) is not None
            }
            if query_nutrients & tag_nutrients:
                conflicts.append(col)
        return list(dict.fromkeys(conflicts))

    def _query_intent_exclusion_columns(
        self,
        query: str | None,
        df: pd.DataFrame,
    ) -> list[str]:
        excluded = [
            *self._resolve_negated_query_tag_columns(query, df),
            *self._resolve_modifier_conflict_columns(query, df),
        ]
        return list(dict.fromkeys(excluded))

    def _apply_query_intent_exclusions(
        self,
        df: pd.DataFrame,
        query: str | None,
    ) -> pd.DataFrame:
        """Apply structured exclusions inferred from query wording."""
        excluded_cols = self._query_intent_exclusion_columns(query, df)
        excluded_cols = [col for col in excluded_cols if col in df.columns]
        if not excluded_cols or df.empty:
            return df

        keep = pd.Series(True, index=df.index)
        for col in excluded_cols:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            keep &= values.ne(1)
        return df.loc[keep].reset_index(drop=True)

    def _semantic_query_text(self, query: str, df: pd.DataFrame | None) -> str:
        """Remove negated tag phrases before embedding the semantic query text."""
        if df is None:
            return query

        tokens = self._query_tokens(query)
        if not tokens:
            return query

        normalized_to_col = self._normalized_tag_map(df)
        if not normalized_to_col:
            return query

        remove_indices: set[int] = set()
        for idx, token in enumerate(tokens):
            if token not in _NEGATION_TOKENS:
                continue
            tag_col, end = self._resolve_tag_from_tokens(tokens, idx + 1, normalized_to_col)
            if tag_col is not None:
                remove_indices.update(range(idx, end))

        if not remove_indices:
            return query

        remaining = [token for idx, token in enumerate(tokens) if idx not in remove_indices]
        return " ".join(remaining) if remaining else "recipe"

    def _candidate_pool_multiplier(self, query: str | None, df: pd.DataFrame) -> int:
        if self._query_intent_exclusion_columns(query, df):
            return _INTENT_FILTER_CANDIDATE_POOL_MULTIPLIER
        return _DEFAULT_CANDIDATE_POOL_MULTIPLIER

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

    def _attach_foodcom_images(self, results: pd.DataFrame) -> pd.DataFrame:
        """Patch missing images in a result frame via Food.com lookup."""
        return attach_foodcom_images(results, fallback_image=FALLBACK_IMAGE_URL)

    def _load_aligned_lexical_index(self, path: Path | None = None) -> LexicalIndex | None:
        """Load optional lexical artifacts and align rows to loaded metadata."""
        if self.metadata is None:
            raise RuntimeError("Metadata must be loaded before lexical artifacts.")

        lexical_path = Path(path) if path else DEFAULT_LEXICAL_INDEX_PATH
        if not lexical_path.exists():
            logger.info(
                "Lexical index not found at %s; using dense-only text search.", lexical_path
            )
            return None

        try:
            index = load_lexical_index(lexical_path)
            aligned = align_lexical_index(index, self.metadata[ID_COLUMN])
        except Exception as exc:
            logger.warning(
                "Unable to load lexical index from %s; using dense-only text search: %s",
                lexical_path,
                exc,
            )
            return None

        if aligned.matrix.shape[0] != len(self.metadata):
            logger.warning(
                "Lexical index row count does not match metadata; using dense-only text search."
            )
            return None

        logger.info(
            "Loaded lexical TF-IDF index: rows=%d, features=%d.",
            aligned.matrix.shape[0],
            aligned.matrix.shape[1],
        )
        return aligned

    def load(
        self,
        *,
        processed_path: Path | None = None,
        embeddings_path: Path | None = None,
        recipe_ids_path: Path | None = None,
        lexical_index_path: Path | None = None,
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
        lexical_index_path:
            Optional override for TF-IDF lexical index path. Missing or stale
            artifacts automatically fall back to dense-only text search.
        """
        metadata = load_processed_recipes(processed_path)
        embeddings = load_embeddings(embeddings_path)
        recipe_ids = load_recipe_ids(recipe_ids_path)

        self.metadata = self._align_metadata_by_recipe_id(metadata, recipe_ids, embeddings)
        self.embeddings = embeddings
        self.one_hot_tag_columns = get_one_hot_tag_columns(self.metadata)
        self._attach_image_urls()
        self._lexical_index = self._load_aligned_lexical_index(lexical_index_path)

        encoder_config = self._get_encoder_config()
        self.encoder = RecipeEncoder(config=encoder_config)
        self.encoder.load()

        logger.info(
            "RetrievalService loaded: %d rows aligned to %d embedding vectors.",
            len(self.metadata),
            self.embeddings.shape[0],
        )
        logger.info(
            "Detected %d one-hot tag columns for retrieval intent matching.",
            len(self.one_hot_tag_columns),
        )

        if self.embeddings.shape[0] < _SMALL_ARTIFACT_WARN_THRESHOLD:
            logger.warning(
                "Embedding artifact is a tiny subset (%d rows). Search quality for broad "
                "queries may be poor. Rebuild embeddings without --limit for full coverage.",
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
        if torch is None or AutoModel is None or AutoProcessor is None:
            raise RuntimeError(
                "Image search requires torch and transformers. "
                "Install project requirements before using SigLIP search."
            )

        try:
            self._siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
            self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(self._siglip_device)
        except ImportError as exc:
            raise RuntimeError(
                "Image search requires the full SigLIP dependency stack, including "
                "sentencepiece. Install project requirements and sentencepiece before "
                "using image search."
            ) from exc
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

        Returns a DataFrame with ['recipe_id', 'x_proj', 'y_proj'] and metadata.
        Returns an empty DataFrame if projections are not loaded.
        """
        if self.metadata is None or "x_proj" not in self.metadata.columns:
            return pd.DataFrame(columns=[ID_COLUMN, "x_proj", "y_proj"])

        # Return only the rows that actually have projection coordinates
        has_proj = self.metadata["x_proj"].notna()
        cols = [ID_COLUMN, "x_proj", "y_proj"]

        cols_to_check = ["name", "minutes", "n_ingredients"]
        from recipe_discovery.data.schema import NUTRITION_COLUMNS

        cols_to_check.extend(NUTRITION_COLUMNS)

        for c in cols_to_check:
            if c in self.metadata.columns:
                cols.append(c)

        df = self.metadata.loc[has_proj, cols].copy()

        kmeans_path = ARTIFACTS_DIR / "kmeans.joblib"
        if kmeans_path.exists():
            try:
                from recipe_discovery.clustering.kmeans import KMeans
                import numpy as np

                model = KMeans.load(kmeans_path)
                if hasattr(model, "labels_") and model.labels_ is not None:
                    if len(model.labels_) == len(self.metadata):
                        labels_array = np.asarray(model.labels_)
                        mask = has_proj.to_numpy()
                        df["cluster"] = [f"Cluster {int(lbl)}" for lbl in labels_array[mask]]
            except Exception as e:
                logger.warning(f"Failed to load clustering model labels: {e}")

        return df.reset_index(drop=True)

    def _empty_search_frame(self, extra_columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Return an empty metadata-shaped result frame with score columns."""
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        result = self.metadata.iloc[0:0].copy()
        for column in extra_columns or []:
            result[column] = pd.Series(dtype=float)
        result["similarity_score"] = pd.Series(dtype=float)
        return result

    def _candidate_pool_size(self, top_k: int, score_count: int, query_text: str) -> int:
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")
        if top_k <= 0:
            return 0
        candidate_pool_multiplier = self._candidate_pool_multiplier(query_text, self.metadata)
        return min(score_count, max(top_k * candidate_pool_multiplier, top_k))

    @staticmethod
    def _top_score_indices(scores: np.ndarray, candidate_pool: int) -> np.ndarray:
        """Return candidate indices sorted by descending score."""
        if candidate_pool <= 0 or len(scores) == 0:
            return np.array([], dtype=int)
        if candidate_pool >= len(scores):
            top_idx = np.arange(len(scores), dtype=int)
        else:
            top_idx = np.argpartition(-scores, candidate_pool - 1)[:candidate_pool]
        return top_idx[np.argsort(-scores[top_idx], kind="stable")]

    @staticmethod
    def _normalize_candidate_scores(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        min_value = float(values.min())
        max_value = float(values.max())
        if np.isclose(max_value, min_value):
            return np.zeros_like(values, dtype=float)
        return (values - min_value) / (max_value - min_value)

    def _rank_candidates_from_scores(
        self,
        request: RetrievalRequest,
        *,
        ranked_idx: np.ndarray,
        scores: np.ndarray,
        query_text: str,
        limit_to_top_k: bool,
        extra_score_columns: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Apply filters, tag boosts, and final truncation to pre-ranked candidates."""
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        if len(ranked_idx) == 0:
            return self._empty_search_frame(
                extra_score_columns.keys() if extra_score_columns else None
            )

        candidates = self.metadata.iloc[ranked_idx].copy()
        if extra_score_columns:
            for column, values in extra_score_columns.items():
                candidates[column] = values[ranked_idx]
        candidates["similarity_score"] = scores[ranked_idx]

        filtered = apply_basic_filters(
            candidates,
            dietary_filter=request.dietary_filter,
            max_time_minutes=request.max_time_minutes,
            max_ingredients=request.max_ingredients,
            max_calories=request.max_calories,
            max_fat=request.max_fat,
            max_sugar=request.max_sugar,
            max_sodium=request.max_sodium,
            max_protein=request.max_protein,
            max_saturated_fat=request.max_saturated_fat,
            max_carbohydrates=request.max_carbohydrates,
            min_rating=request.min_rating,
        )

        filtered = self._apply_query_intent_exclusions(filtered, query_text)
        if filtered.empty:
            return filtered

        matched_tag_col = self._resolve_query_tag_column(query_text, filtered)
        if matched_tag_col and matched_tag_col in filtered.columns:
            ranked = filtered.copy()
            ranked["query_tag_match"] = (
                pd.to_numeric(ranked[matched_tag_col], errors="coerce")
                .fillna(0)
                .astype(int)
                .eq(1)
                .astype(int)
            )
            ranked["matched_query_tag"] = matched_tag_col
            ranked["boosted_similarity_score"] = (
                ranked["similarity_score"] + _EXACT_TAG_SCORE_BOOST * ranked["query_tag_match"]
            )
            ranked = ranked.sort_values(
                ["boosted_similarity_score", "similarity_score"],
                ascending=False,
                kind="mergesort",
            )
        else:
            ranked = filtered.sort_values(
                "similarity_score",
                ascending=False,
                kind="mergesort",
            )

        if limit_to_top_k:
            ranked = ranked.head(request.top_k)

        return ranked.reset_index(drop=True)

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
            return self._empty_search_frame()

        scores = cosine_similarity(query=query_vec, matrix=embeddings)

        query_text = request.query or ""
        candidate_pool = self._candidate_pool_size(request.top_k, len(scores), query_text)
        if candidate_pool == 0:
            return self._empty_search_frame()

        # Candidate pooling prevents filters from exhausting a too-small pre-truncated set.
        ranked_idx = self._top_score_indices(scores, candidate_pool)
        return self._rank_candidates_from_scores(
            request,
            ranked_idx=ranked_idx,
            scores=scores,
            query_text=query_text,
            limit_to_top_k=limit_to_top_k,
        )

    def _search_hybrid_candidates(
        self,
        request: RetrievalRequest,
        *,
        query_vec: np.ndarray,
        limit_to_top_k: bool,
    ) -> pd.DataFrame:
        """Return text matches using a dense + TF-IDF keyword candidate union."""
        if self.embeddings is None or self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")
        if self._lexical_index is None:
            return self._search_candidates_for_vector(
                request,
                query_vec=query_vec,
                embeddings=self.embeddings,
                limit_to_top_k=limit_to_top_k,
            )

        extra_columns = ["semantic_similarity_score", "keyword_similarity_score"]
        if request.top_k <= 0:
            return self._empty_search_frame(extra_columns)

        query_text = request.query or ""
        semantic_query = self._semantic_query_text(query_text, self.metadata)
        semantic_scores = cosine_similarity(query=query_vec, matrix=self.embeddings)
        keyword_scores = score_lexical_query(self._lexical_index, semantic_query)
        if len(keyword_scores) != len(semantic_scores):
            logger.warning("Lexical and dense score counts differ; using dense-only text search.")
            return self._search_candidates_for_vector(
                request,
                query_vec=query_vec,
                embeddings=self.embeddings,
                limit_to_top_k=limit_to_top_k,
            )

        candidate_pool = self._candidate_pool_size(request.top_k, len(semantic_scores), query_text)
        if candidate_pool == 0:
            return self._empty_search_frame(extra_columns)

        semantic_idx = self._top_score_indices(semantic_scores, candidate_pool)
        keyword_idx = self._top_score_indices(keyword_scores, candidate_pool)
        candidate_idx = np.unique(np.concatenate([semantic_idx, keyword_idx])).astype(int)
        if candidate_idx.size == 0:
            return self._empty_search_frame(extra_columns)

        normalized_semantic = self._normalize_candidate_scores(semantic_scores[candidate_idx])
        normalized_keyword = self._normalize_candidate_scores(keyword_scores[candidate_idx])
        combined_candidate_scores = (
            _HYBRID_SEMANTIC_WEIGHT * normalized_semantic
            + _HYBRID_KEYWORD_WEIGHT * normalized_keyword
        )

        combined_scores = np.zeros_like(semantic_scores, dtype=float)
        combined_scores[candidate_idx] = combined_candidate_scores
        ranked_idx = candidate_idx[np.argsort(-combined_candidate_scores, kind="stable")]
        return self._rank_candidates_from_scores(
            request,
            ranked_idx=ranked_idx,
            scores=combined_scores,
            query_text=query_text,
            limit_to_top_k=limit_to_top_k,
            extra_score_columns={
                "semantic_similarity_score": semantic_scores,
                "keyword_similarity_score": keyword_scores,
            },
        )

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

        query_vec = self._encode_text_query_vector(request.query)
        return self._search_hybrid_candidates(
            request,
            query_vec=query_vec,
            limit_to_top_k=limit_to_top_k,
        )

    def _encode_text_query_vector(self, query: str) -> np.ndarray:
        """Encode and cache a normalized text query vector for this service."""
        if self.encoder is None:
            raise RuntimeError("RetrievalService is not loaded.")
        if not query or not query.strip():
            raise ValueError("Search query is required for text feedback.")

        semantic_query = self._semantic_query_text(query, self.metadata)
        cache_key = self._normalize_token(semantic_query)
        cached = self._text_query_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        encoded = self.encoder.encode([semantic_query], show_progress=False)
        vec = np.asarray(encoded, dtype=float)[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._text_query_cache[cache_key] = vec
        return vec.copy()

    def encode_text_query(self, query: str) -> np.ndarray:
        """Encode a text query into the loaded SBERT embedding space."""
        return self._encode_text_query_vector(query)

    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        """Encode an uploaded image into the loaded SigLIP embedding space."""
        self._load_siglip_embeddings()
        return self._encode_image(image)

    def encode_combined_query(
        self,
        text: str,
        image: Image.Image,
        *,
        alpha: float = SIGLIP_COMBINED_DEFAULT_IMAGE_WEIGHT,
    ) -> np.ndarray:
        """Encode text + image into the loaded SigLIP embedding space."""
        self._load_siglip_embeddings()
        semantic_text = self._semantic_query_text(text, self.metadata)
        return self.encode_combined(semantic_text, image, alpha=alpha)

    def search_with_negative_feedback(
        self,
        request: RetrievalRequest,
        *,
        query_vec: np.ndarray,
        negative_recipe_ids: set[str],
        alpha: float = 0.3,
        embedding_space: str = "text",
    ) -> pd.DataFrame:
        """Run retrieval after applying negative Rocchio feedback."""
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        if embedding_space == "text":
            if self.embeddings is None:
                raise RuntimeError("RetrievalService is not loaded.")
            embeddings = self.embeddings
        elif embedding_space == "siglip":
            self._load_siglip_embeddings()
            if self._siglip_embeddings is None:
                raise RuntimeError("SigLIP embeddings failed to load.")
            embeddings = self._siglip_embeddings
        else:
            raise ValueError("embedding_space must be 'text' or 'siglip'.")

        excluded = {
            str(recipe_id).strip() for recipe_id in negative_recipe_ids if str(recipe_id).strip()
        }
        if not excluded:
            results = self._search_candidates_for_vector(
                request,
                query_vec=np.asarray(query_vec, dtype=float),
                embeddings=embeddings,
                limit_to_top_k=True,
            )
            return self._attach_foodcom_images(results)

        metadata_ids = self._normalize_recipe_ids(self.metadata[ID_COLUMN])
        excluded_series = pd.Series(list(excluded), dtype=str)
        excluded = set(self._normalize_recipe_ids(excluded_series).tolist())

        negative_mask = metadata_ids.isin(excluded).to_numpy()
        if not negative_mask.any():
            results = self._search_candidates_for_vector(
                request,
                query_vec=np.asarray(query_vec, dtype=float),
                embeddings=embeddings,
                limit_to_top_k=True,
            )
            return self._attach_foodcom_images(results)

        query_vec = np.asarray(query_vec, dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        negative_vectors = embeddings[negative_mask]
        mean_negative = negative_vectors.mean(axis=0)
        adjusted = query_vec - alpha * mean_negative
        adjusted_norm = np.linalg.norm(adjusted)
        if adjusted_norm > 0:
            adjusted = adjusted / adjusted_norm
        else:
            adjusted = query_vec

        original_top_k = request.top_k
        feedback_request = replace(
            request,
            top_k=max(original_top_k + len(excluded) + 10, original_top_k),
        )
        candidates = self._search_candidates_for_vector(
            feedback_request,
            query_vec=adjusted,
            embeddings=embeddings,
            limit_to_top_k=False,
        )

        if not candidates.empty:
            candidate_ids = self._normalize_recipe_ids(candidates[ID_COLUMN])
            candidates = candidates.loc[~candidate_ids.isin(excluded)].copy()

        results = candidates.head(original_top_k).reset_index(drop=True)
        return self._attach_foodcom_images(results)

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
        results = reranked.head(request.top_k).reset_index(drop=True)
        return self._attach_foodcom_images(results)

    def search(self, request: RetrievalRequest) -> pd.DataFrame:
        """Return filtered top-k recipe matches for a query.

        This v1 runtime path intentionally performs direct cosine scoring
        against the loaded embedding matrix before candidate-pool filtering.
        """
        results = self._search_candidates(request, limit_to_top_k=True)
        return self._attach_foodcom_images(results)

    def search_by_image(self, image: Image.Image, request: RetrievalRequest) -> pd.DataFrame:
        """Return filtered top-k recipe matches for an uploaded image."""
        if self.metadata is None:
            raise RuntimeError("RetrievalService is not loaded.")

        self._load_siglip_embeddings()
        if self._siglip_embeddings is None:
            raise RuntimeError("SigLIP embeddings failed to load.")

        query_vec = self._encode_image(image)
        results = self._search_candidates_for_vector(
            request,
            query_vec=query_vec,
            embeddings=self._siglip_embeddings,
            limit_to_top_k=True,
        )
        return self._attach_foodcom_images(results)

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
        semantic_text = self._semantic_query_text(text, self.metadata)
        text_vec = self._encode_siglip_text(semantic_text)
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
        filtered = self._apply_query_intent_exclusions(filtered, request.query or text)

        if filtered.empty:
            return filtered

        ranked = filtered.sort_values("similarity_score", ascending=False)
        results = ranked.head(request.top_k).reset_index(drop=True)
        return self._attach_foodcom_images(results)
