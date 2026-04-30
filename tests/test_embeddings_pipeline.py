"""Robust tests for embedding generation and artifact persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recipe_discovery.data.corpus import build_corpus
from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.schema import ID_COLUMN
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder
from recipe_discovery.embeddings.store import (
    load_embeddings,
    load_recipe_ids,
    save_embeddings,
    save_recipe_ids,
)


class _DummyModel:
    """Deterministic embedding backend used for tests."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        _ = (batch_size, convert_to_numpy, show_progress_bar)
        vectors: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=float)
            for idx, byte in enumerate(str(text).encode("utf-8")):
                vec[idx % self._dim] += (byte % 31) + 1
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)



def _encode_texts(texts: list[str], *, dim: int = 16) -> np.ndarray:
    encoder = RecipeEncoder(config=EmbeddingConfig(batch_size=8, normalize=True))
    encoder.model = _DummyModel(dim=dim)
    return encoder.encode(texts, show_progress=False)


@pytest.fixture(scope="module")
def processed_subset() -> pd.DataFrame:
    """Small real-data subset used for fast integration checks."""
    return load_processed_recipes().head(64).copy()



def test_embeddings_row_count_matches_processed_row_count(processed_subset: pd.DataFrame) -> None:
    corpus_df = build_corpus(processed_subset)
    embeddings = _encode_texts(corpus_df["recipe_text"].tolist())

    assert embeddings.shape[0] == len(corpus_df)



def test_recipe_ids_row_count_matches_embeddings_row_count(processed_subset: pd.DataFrame) -> None:
    corpus_df = build_corpus(processed_subset)
    embeddings = _encode_texts(corpus_df["recipe_text"].tolist())
    recipe_ids = corpus_df[ID_COLUMN]

    assert len(recipe_ids) == embeddings.shape[0]



def test_embeddings_artifacts_saved_and_reloadable(
    processed_subset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    corpus_df = build_corpus(processed_subset)
    embeddings = _encode_texts(corpus_df["recipe_text"].tolist())
    recipe_ids = corpus_df[ID_COLUMN]

    emb_path = save_embeddings(embeddings, tmp_path / "recipe_embeddings.npy")
    ids_path = save_recipe_ids(recipe_ids, tmp_path / "recipe_ids.csv")

    loaded_embeddings = load_embeddings(emb_path)
    loaded_ids = load_recipe_ids(ids_path)

    assert loaded_embeddings.shape == embeddings.shape
    assert np.allclose(loaded_embeddings, embeddings)
    assert loaded_ids.astype(str).tolist() == recipe_ids.astype(str).tolist()



def test_empty_or_partially_missing_text_rows_do_not_crash_embedding_generation() -> None:
    sample = pd.DataFrame(
        {
            "recipe_id": [1, 2, 3],
            "name": ["", None, "Demo"],
            "description": [None, "", "Simple"],
            "ingredients": [None, "salt", ""],
            "steps": ["", None, "mix"],
            "minutes": [10, 20, np.nan],
            "n_steps": [1, np.nan, 2],
            "n_ingredients": [2, 3, np.nan],
            "vegetarian": [1, 0, 1],
        }
    )

    corpus_df = build_corpus(sample)
    embeddings = _encode_texts(corpus_df["recipe_text"].tolist())

    assert embeddings.shape == (3, 16)



def test_repeated_runs_preserve_shape_and_artifact_consistency(
    processed_subset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    corpus_df = build_corpus(processed_subset)
    texts = corpus_df["recipe_text"].tolist()
    ids = corpus_df[ID_COLUMN]

    embeddings_a = _encode_texts(texts)
    embeddings_b = _encode_texts(texts)

    assert embeddings_a.shape == embeddings_b.shape
    assert np.allclose(embeddings_a, embeddings_b)

    emb_path_a = save_embeddings(embeddings_a, tmp_path / "run_a_embeddings.npy")
    emb_path_b = save_embeddings(embeddings_b, tmp_path / "run_b_embeddings.npy")
    ids_path_a = save_recipe_ids(ids, tmp_path / "run_a_recipe_ids.csv")
    ids_path_b = save_recipe_ids(ids, tmp_path / "run_b_recipe_ids.csv")

    loaded_a = load_embeddings(emb_path_a)
    loaded_b = load_embeddings(emb_path_b)
    loaded_ids_a = load_recipe_ids(ids_path_a)
    loaded_ids_b = load_recipe_ids(ids_path_b)

    assert loaded_a.shape == loaded_b.shape
    assert np.allclose(loaded_a, loaded_b)
    assert loaded_ids_a.astype(str).tolist() == loaded_ids_b.astype(str).tolist()



def test_missing_embedding_artifact_path_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_embeddings.npy"
    with pytest.raises(FileNotFoundError, match="missing_embeddings"):
        load_embeddings(missing)



def test_missing_recipe_ids_artifact_path_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_recipe_ids.csv"
    with pytest.raises(FileNotFoundError, match="missing_recipe_ids"):
        load_recipe_ids(missing)
