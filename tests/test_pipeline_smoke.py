"""Mini end-to-end smoke test for data, embeddings, index, and retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recipe_discovery.data.corpus import build_corpus
from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.schema import ID_COLUMN
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder
from recipe_discovery.embeddings.index import build_index, load_index, save_index
from recipe_discovery.embeddings.store import (
    load_embeddings,
    load_recipe_ids,
    save_embeddings,
    save_recipe_ids,
)
from recipe_discovery.retrieval.service import RetrievalRequest, RetrievalService


def _stable_vector(text: str, dim: int = 24) -> np.ndarray:
    """Create a deterministic dense vector from text bytes."""
    vec = np.zeros(dim, dtype=float)
    for idx, byte in enumerate(str(text).encode("utf-8")):
        vec[idx % dim] += (byte % 29) + 1
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class _DummyModel:
    """Deterministic model used to avoid network/model download in smoke tests."""

    def __init__(self, dim: int = 24) -> None:
        self._dim = dim

    def get_embedding_dimension(self) -> int:
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
        vectors = [_stable_vector(text, dim=self._dim) for text in texts]
        matrix = np.vstack(vectors)
        if not normalize_embeddings:
            return matrix
        return matrix


class _FakeServiceEncoder:
    """Fake encoder with RetrievalService-compatible interface."""

    def __init__(self, config: object) -> None:
        self.config = config

    def load(self) -> None:
        return None

    def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        _ = show_progress
        return np.vstack([_stable_vector(text, dim=24) for text in texts])



def test_end_to_end_mini_pipeline_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1) load processed data
    processed = load_processed_recipes().head(50).copy()

    # Ensure the filter path has at least one guaranteed passing row.
    for col in ["vegetarian", "vegan", "gluten-free"]:
        if col not in processed.columns:
            processed[col] = 0
    processed.loc[processed.index[0], "vegetarian"] = 1
    processed.loc[processed.index[0], "vegan"] = 1
    processed.loc[processed.index[0], "gluten-free"] = 1
    if "minutes" in processed.columns:
        processed.loc[processed.index[0], "minutes"] = 15
    if "n_ingredients" in processed.columns:
        processed.loc[processed.index[0], "n_ingredients"] = 4

    # 2) build corpus
    corpus_df = build_corpus(processed)
    assert "recipe_text" in corpus_df.columns
    assert len(corpus_df) == len(processed)

    # 3) build embeddings with deterministic model
    encoder = RecipeEncoder(config=EmbeddingConfig(batch_size=8, normalize=True))
    encoder.model = _DummyModel(dim=24)
    embeddings = encoder.encode(corpus_df["recipe_text"].tolist(), show_progress=False)

    # 4) save artifacts and build index
    emb_path = save_embeddings(embeddings, tmp_path / "recipe_embeddings.npy")
    ids_path = save_recipe_ids(corpus_df[ID_COLUMN], tmp_path / "recipe_ids.csv")

    index = build_index(embeddings, n_neighbors=5)
    index_path = save_index(index, tmp_path / "recipe_index.joblib")
    loaded_index = load_index(index_path)

    distances, neighbors = loaded_index.kneighbors(embeddings[:1], n_neighbors=3)
    assert distances.shape == (1, 3)
    assert neighbors.shape == (1, 3)

    # 5) load retrieval service using these in-test artifacts
    monkeypatch.setattr("recipe_discovery.retrieval.service.RecipeEncoder", _FakeServiceEncoder)
    monkeypatch.setattr(
        "recipe_discovery.retrieval.service.load_processed_recipes",
        lambda path=None: corpus_df.copy(),
    )
    monkeypatch.setattr(
        "recipe_discovery.retrieval.service.load_embeddings",
        lambda path=None: load_embeddings(emb_path),
    )
    monkeypatch.setattr(
        "recipe_discovery.retrieval.service.load_recipe_ids",
        lambda path=None: load_recipe_ids(ids_path),
    )

    service = RetrievalService()
    service.load()

    assert service.encoder is not None
    assert service.embeddings is not None
    assert service.metadata is not None
    assert len(service.metadata) == service.embeddings.shape[0]

    # 6) run multiple example queries and verify result shape/columns
    requests = [
        RetrievalRequest(query="quick vegetarian dinner", top_k=5),
        RetrievalRequest(query="easy vegan meal", top_k=5, dietary_filter="vegan"),
        RetrievalRequest(
            query="gluten free quick",
            top_k=5,
            dietary_filter="gluten-free",
            max_time_minutes=20,
            max_ingredients=6,
        ),
    ]

    for req in requests:
        result = service.search(req)
        assert "similarity_score" in result.columns
        assert len(result) <= req.top_k

    filtered = service.search(
        RetrievalRequest(
            query="quick healthy",
            top_k=10,
            dietary_filter="vegetarian",
            max_time_minutes=20,
            max_ingredients=6,
        )
    )

    assert not filtered.empty
    assert (filtered["vegetarian"] == 1).all()
    assert (filtered["minutes"] <= 20).all()
    assert (filtered["n_ingredients"] <= 6).all()
