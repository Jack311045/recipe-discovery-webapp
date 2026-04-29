"""Tests for retrieval integration and utility behavior."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.schema import get_one_hot_tag_columns
from recipe_discovery.embeddings.index import build_index, load_index, save_index
from recipe_discovery.embeddings.store import (
    load_embeddings,
    load_recipe_ids,
    save_embeddings,
    save_recipe_ids,
)
from recipe_discovery.models.regression import RecipeRegressor
from recipe_discovery.retrieval.service import RetrievalRequest, RetrievalService
from recipe_discovery.retrieval.similarity import cosine_similarity


@pytest.fixture(scope="module")
def processed_df() -> pd.DataFrame:
    """Load the real processed CSV once for schema-level assertions."""
    return load_processed_recipes()


@pytest.fixture(scope="module")
def artifact_bundle() -> tuple[np.ndarray, pd.Series]:
    """Load saved embedding artifacts once for integration checks."""
    return load_embeddings(), load_recipe_ids()


@pytest.fixture
def toy_service() -> RetrievalService:
    """Create a minimal loaded service for deterministic search/filter tests."""

    class DummyEncoder:
        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            _ = show_progress
            vectors: list[np.ndarray] = []
            for text in texts:
                if "alt" in text.lower():
                    vectors.append(np.array([0.0, 1.0]))
                else:
                    vectors.append(np.array([1.0, 0.0]))
            return np.vstack(vectors)

    service = RetrievalService()
    service.encoder = DummyEncoder()
    service.embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.85, 0.15],
            [0.7, 0.3],
            [0.0, 1.0],
        ]
    )
    service.metadata = pd.DataFrame(
        {
            "recipe_id": ["1", "2", "3", "4", "5"],
            "name": ["A", "B", "C", "D", "E"],
            "minutes": [10, 20, 25, 35, 50],
            "n_ingredients": [8, 4, 5, 3, 2],
            "vegetarian": [0, 0, 1, 1, 0],
            "vegan": [0, 0, 0, 1, 1],
            "gluten-free": [0, 0, 1, 0, 1],
            "korean": [0, 0, 0, 1, 0],
            "italian": [1, 0, 0, 0, 0],
            "desserts": [0, 0, 1, 0, 0],
            "5-ingredients-or-less": [0, 1, 1, 1, 1],
            "15-minutes-or-less": [1, 0, 0, 0, 0],
            "30-minutes-or-less": [1, 1, 1, 0, 0],
            "60-minutes-or-less": [1, 1, 1, 1, 1],
        }
    )
    return service


def _train_tiny_regressor(df: pd.DataFrame, feature_columns: list[str]) -> RecipeRegressor:
    x = df.loc[:, feature_columns].to_numpy(dtype=float)
    y = (
        0.02 * df["minutes"].to_numpy(dtype=float)
        + 0.15 * df["n_ingredients"].to_numpy(dtype=float)
    )
    model = RecipeRegressor(alpha=1.0)
    model.fit(x, y)
    return model


def test_processed_csv_loads_successfully(processed_df: pd.DataFrame) -> None:
    required = {"recipe_id", "name", "description", "ingredients", "steps", "minutes"}
    assert not processed_df.empty
    assert required.issubset(set(processed_df.columns))


def test_one_hot_tag_columns_detected_correctly(processed_df: pd.DataFrame) -> None:
    one_hot_cols = get_one_hot_tag_columns(processed_df)
    assert one_hot_cols
    assert "vegetarian" in one_hot_cols
    assert "vegan" in one_hot_cols


def test_embeddings_artifacts_can_be_loaded(
    artifact_bundle: tuple[np.ndarray, pd.Series],
) -> None:
    embeddings, recipe_ids = artifact_bundle
    assert embeddings.ndim == 2
    assert embeddings.shape[0] > 0
    assert len(recipe_ids) == embeddings.shape[0]


def test_retrieval_service_loads_without_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
    artifact_bundle: tuple[np.ndarray, pd.Series],
) -> None:
    class FakeRecipeEncoder:
        def __init__(self, config: object) -> None:
            self.config = config

        def load(self) -> None:
            return None

        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            _ = show_progress
            return np.zeros((len(texts), 384), dtype=float)

    monkeypatch.setattr("recipe_discovery.retrieval.service.RecipeEncoder", FakeRecipeEncoder)

    service = RetrievalService()
    service.load()

    embeddings, recipe_ids = artifact_bundle
    assert service.embeddings is not None
    assert service.metadata is not None
    assert service.encoder is not None
    assert service.embeddings.shape == embeddings.shape
    assert len(service.metadata) == service.embeddings.shape[0]
    actual_ids = service.metadata["recipe_id"].astype(str).str.replace(r"\.0+$", "", regex=True)
    expected_ids = recipe_ids.astype(str).str.replace(r"\.0+$", "", regex=True)
    assert actual_ids.tolist() == expected_ids.tolist()


def test_retrieval_service_load_accepts_path_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRecipeEncoder:
        def __init__(self, config: object) -> None:
            self.config = config

        def load(self) -> None:
            return None

    monkeypatch.setattr("recipe_discovery.retrieval.service.RecipeEncoder", FakeRecipeEncoder)

    processed = pd.DataFrame(
        {
            "recipe_id": ["10", "20", "30"],
            "name": ["A", "B", "C"],
            "description": ["d1", "d2", "d3"],
            "ingredients": ["i1", "i2", "i3"],
            "steps": ["s1", "s2", "s3"],
            "minutes": [10, 20, 30],
            "n_steps": [1, 1, 1],
            "n_ingredients": [2, 3, 4],
            "vegetarian": [1, 0, 1],
        }
    )
    processed_path = tmp_path / "processed_subset.csv"
    processed.to_csv(processed_path, index=False)

    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=float)
    emb_path = save_embeddings(embeddings, tmp_path / "recipe_embeddings.npy")
    ids_path = save_recipe_ids(pd.Series(["10", "20", "30"]), tmp_path / "recipe_ids.csv")

    service = RetrievalService()
    service.load(
        processed_path=processed_path,
        embeddings_path=emb_path,
        recipe_ids_path=ids_path,
    )

    assert service.encoder is not None
    assert service.embeddings is not None
    assert service.metadata is not None
    assert len(service.metadata) == service.embeddings.shape[0] == 3
    assert service.metadata["recipe_id"].astype(str).tolist() == ["10", "20", "30"]


def test_load_warns_for_tiny_embedding_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeRecipeEncoder:
        def __init__(self, config: object) -> None:
            self.config = config

        def load(self) -> None:
            return None

        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            _ = show_progress
            return np.zeros((len(texts), 4), dtype=float)

    metadata = pd.DataFrame(
        {
            "recipe_id": [str(i) for i in range(10)],
            "name": [f"recipe-{i}" for i in range(10)],
            "description": ["d"] * 10,
            "ingredients": ["i"] * 10,
            "steps": ["s"] * 10,
            "minutes": [20] * 10,
            "n_steps": [3] * 10,
            "n_ingredients": [6] * 10,
            "vegan": [0, 1] * 5,
        }
    )
    embeddings = np.random.default_rng(42).normal(size=(10, 4)).astype(np.float32)
    recipe_ids = pd.Series([str(i) for i in range(10)], name="recipe_id")

    monkeypatch.setattr("recipe_discovery.retrieval.service.RecipeEncoder", FakeRecipeEncoder)
    monkeypatch.setattr("recipe_discovery.retrieval.service.load_processed_recipes", lambda _: metadata)
    monkeypatch.setattr("recipe_discovery.retrieval.service.load_embeddings", lambda _: embeddings)
    monkeypatch.setattr("recipe_discovery.retrieval.service.load_recipe_ids", lambda _: recipe_ids)

    service = RetrievalService()

    with caplog.at_level(logging.WARNING):
        service.load(
            processed_path=tmp_path / "metadata.csv",
            embeddings_path=tmp_path / "embeddings.npy",
            recipe_ids_path=tmp_path / "ids.csv",
        )

    assert any("tiny subset" in record.getMessage() for record in caplog.records)


def test_metadata_alignment_uses_recipe_id_not_accidental_row_order() -> None:
    service = RetrievalService()
    metadata = pd.DataFrame(
        {
            "recipe_id": ["30", "10", "20"],
            "name": ["C", "A", "B"],
            "minutes": [30, 10, 20],
        }
    )
    recipe_ids = pd.Series(["10", "20", "30"])
    embeddings = np.ones((3, 4), dtype=float)

    aligned = service._align_metadata_by_recipe_id(metadata, recipe_ids, embeddings)

    assert aligned["recipe_id"].tolist() == ["10", "20", "30"]
    assert aligned["name"].tolist() == ["A", "B", "C"]


def test_text_query_returns_ranked_results(toy_service: RetrievalService) -> None:
    result = toy_service.search(RetrievalRequest(query="quick dinner", top_k=3))
    assert len(result) == 3
    assert "similarity_score" in result.columns
    assert result["similarity_score"].is_monotonic_decreasing


def test_exact_tag_query_prioritizes_korean_matches(toy_service: RetrievalService) -> None:
    result = toy_service.search(RetrievalRequest(query="korean", top_k=5))

    assert not result.empty
    assert "query_tag_match" in result.columns
    assert "matched_query_tag" in result.columns
    assert result.iloc[0]["korean"] == 1
    assert result["query_tag_match"].is_monotonic_decreasing


def test_near_exact_tag_query_with_suffix_prioritizes_matches(
    toy_service: RetrievalService,
) -> None:
    result = toy_service.search(RetrievalRequest(query="korean recipes", top_k=5))

    assert not result.empty
    assert "query_tag_match" in result.columns
    assert result.iloc[0]["korean"] == 1


def test_exact_tag_query_prioritizes_vegan_matches(toy_service: RetrievalService) -> None:
    result = toy_service.search(RetrievalRequest(query="vegan", top_k=5))

    assert not result.empty
    assert "query_tag_match" in result.columns
    assert (result.head(2)["vegan"] == 1).all()


def test_exact_tag_query_prioritizes_desserts_matches(toy_service: RetrievalService) -> None:
    result = toy_service.search(RetrievalRequest(query="desserts", top_k=5))

    assert not result.empty
    assert "query_tag_match" in result.columns
    assert result.iloc[0]["desserts"] == 1


def test_free_text_non_exact_query_keeps_semantic_path(
    toy_service: RetrievalService,
) -> None:
    result = toy_service.search(RetrievalRequest(query="quick vegetarian dinner", top_k=5))

    assert not result.empty
    assert "similarity_score" in result.columns
    assert "query_tag_match" not in result.columns


def test_unknown_query_does_not_crash_and_falls_back(
    toy_service: RetrievalService,
) -> None:
    result = toy_service.search(RetrievalRequest(query="totally-unknown-cuisine", top_k=5))

    assert not result.empty
    assert "similarity_score" in result.columns
    assert "query_tag_match" not in result.columns


def test_dietary_filter_works_on_one_hot_columns(toy_service: RetrievalService) -> None:
    result = toy_service.search(
        RetrievalRequest(query="quick dinner", top_k=1, dietary_filter="vegetarian")
    )
    assert len(result) == 1
    assert int(result.iloc[0]["vegetarian"]) == 1
    assert result.iloc[0]["recipe_id"] == "3"


def test_dietary_filter_supports_gluten_free_label(toy_service: RetrievalService) -> None:
    result = toy_service.search(
        RetrievalRequest(query="quick dinner", top_k=2, dietary_filter="gluten free")
    )

    assert not result.empty
    assert (result["gluten-free"] == 1).all()


def test_max_time_minutes_filter_works(toy_service: RetrievalService) -> None:
    result = toy_service.search(
        RetrievalRequest(query="quick dinner", top_k=5, max_time_minutes=20)
    )
    assert not result.empty
    assert (result["minutes"] <= 20).all()


def test_max_ingredients_filter_works(toy_service: RetrievalService) -> None:
    result = toy_service.search(RetrievalRequest(query="quick dinner", top_k=5, max_ingredients=4))
    assert not result.empty
    assert (result["n_ingredients"] <= 4).all()


def test_empty_result_cases_do_not_crash(toy_service: RetrievalService) -> None:
    result = toy_service.search(
        RetrievalRequest(
            query="quick dinner",
            top_k=5,
            dietary_filter="vegan",
            max_time_minutes=10,
            max_ingredients=1,
        )
    )
    assert result.empty
    assert "similarity_score" in result.columns


def test_candidate_pool_then_filtering_finds_result_beyond_top_k() -> None:
    class DummyEncoder:
        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            _ = (texts, show_progress)
            return np.array([[1.0, 0.0]], dtype=float)

    service = RetrievalService()
    service.encoder = DummyEncoder()
    service.embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.96, 0.04],
            [0.4, 0.6],
            [0.3, 0.7],
        ]
    )
    service.metadata = pd.DataFrame(
        {
            "recipe_id": ["1", "2", "3", "4", "5"],
            "name": ["A", "B", "C", "D", "E"],
            "minutes": [20, 20, 20, 20, 20],
            "n_ingredients": [8, 8, 8, 8, 8],
            "vegetarian": [0, 0, 1, 0, 0],
        }
    )

    result = service.search(
        RetrievalRequest(query="query", top_k=1, dietary_filter="vegetarian")
    )

    assert len(result) == 1
    assert result.iloc[0]["recipe_id"] == "3"


def test_combined_search_uses_text_to_refine_image_candidates() -> None:
    service = RetrievalService()
    visually_relevant_count = 50
    service.metadata = pd.DataFrame(
        {
            "recipe_id": [str(i) for i in range(visually_relevant_count + 1)],
            "name": [f"visual match {i}" for i in range(visually_relevant_count)]
            + ["generic rice dish"],
            "minutes": [20] * (visually_relevant_count + 1),
            "n_ingredients": [5] * (visually_relevant_count + 1),
        }
    )
    service._siglip_embeddings = np.vstack(
        [
            np.tile(np.array([[0.9, 0.1]], dtype=float), (visually_relevant_count, 1)),
            np.array([[0.2, 0.98]], dtype=float),
        ]
    )
    service._encode_image = lambda image: np.array([1.0, 0.0], dtype=float)  # type: ignore[method-assign]
    service._encode_siglip_text = lambda text: np.array([0.0, 1.0], dtype=float)  # type: ignore[method-assign]

    result = service.search_combined(
        "served with rice",
        Image.new("RGB", (1, 1)),
        RetrievalRequest(query="served with rice", top_k=1),
        alpha=0.1,
    )

    assert result.iloc[0]["name"] != "generic rice dish"
    assert "image_similarity_score" in result.columns
    assert "text_similarity_score" in result.columns


def test_search_with_optional_rerank_falls_back_to_similarity_only(
    toy_service: RetrievalService,
) -> None:
    request = RetrievalRequest(query="quick dinner", top_k=4)
    base = toy_service.search(request)
    fallback = toy_service.search_with_optional_rerank(request)

    assert fallback["recipe_id"].tolist() == base["recipe_id"].tolist()
    np.testing.assert_allclose(
        fallback["similarity_score"].to_numpy(dtype=float),
        base["similarity_score"].to_numpy(dtype=float),
    )
    assert "combined_score" in fallback.columns
    assert "predicted_rating" not in fallback.columns
    assert fallback["similarity_score"].is_monotonic_decreasing


def test_rerank_candidates_no_model_does_not_crash(
    toy_service: RetrievalService,
) -> None:
    candidates = toy_service.search(RetrievalRequest(query="quick dinner", top_k=4))
    reranked = toy_service.rerank_candidates(
        candidates,
        regression_model=None,
        feature_columns=None,
    )

    assert not reranked.empty
    assert "combined_score" in reranked.columns
    assert "predicted_rating" not in reranked.columns


def test_rerank_candidates_adds_prediction_columns_and_preserves_similarity(
    toy_service: RetrievalService,
) -> None:
    feature_columns = ["minutes", "n_ingredients"]
    model = _train_tiny_regressor(toy_service.metadata, feature_columns)
    candidates = toy_service.search(RetrievalRequest(query="quick dinner", top_k=4))

    reranked_a = toy_service.rerank_candidates(
        candidates,
        regression_model=model,
        feature_columns=feature_columns,
        similarity_weight=0.7,
        rating_weight=0.3,
    )
    reranked_b = toy_service.rerank_candidates(
        candidates,
        regression_model=model,
        feature_columns=feature_columns,
        similarity_weight=0.7,
        rating_weight=0.3,
    )

    pd.testing.assert_frame_equal(reranked_a, reranked_b)
    assert {"similarity_score", "predicted_rating", "combined_score"}.issubset(
        set(reranked_a.columns)
    )
    assert sorted(reranked_a["similarity_score"].tolist()) == sorted(
        candidates["similarity_score"].tolist()
    )


def test_rerank_candidates_missing_feature_columns_fails_clearly(
    toy_service: RetrievalService,
) -> None:
    feature_columns = ["minutes", "n_ingredients"]
    model = _train_tiny_regressor(toy_service.metadata, feature_columns)
    candidates = toy_service.search(RetrievalRequest(query="quick dinner", top_k=4)).drop(
        columns=["minutes"]
    )

    with pytest.raises(ValueError, match="Missing feature columns"):
        toy_service.rerank_candidates(
            candidates,
            regression_model=model,
            feature_columns=feature_columns,
        )


def test_retrieval_service_loads_regression_artifact_and_scores_candidates(
    toy_service: RetrievalService,
    tmp_path: Path,
) -> None:
    feature_columns = ["minutes", "n_ingredients"]
    model = _train_tiny_regressor(toy_service.metadata, feature_columns)
    model_path = tmp_path / "regressor.joblib"
    model.save(model_path)

    loaded_model = RetrievalService.load_regression_model(model_path)
    assert loaded_model is not None

    candidates = toy_service.search(RetrievalRequest(query="quick dinner", top_k=4))
    reranked = toy_service.rerank_candidates(
        candidates,
        regression_model=loaded_model,
        feature_columns=feature_columns,
    )
    assert "predicted_rating" in reranked.columns
    assert len(reranked) == len(candidates)


def test_retrieval_service_regression_model_roundtrip_stable_for_ranking_use(
    toy_service: RetrievalService,
    tmp_path: Path,
) -> None:
    feature_columns = ["minutes", "n_ingredients"]
    model = _train_tiny_regressor(toy_service.metadata, feature_columns)
    model_path = tmp_path / "regressor.joblib"
    model.save(model_path)
    loaded = RetrievalService.load_regression_model(model_path)
    assert loaded is not None

    candidates = toy_service.search(RetrievalRequest(query="quick dinner", top_k=4))
    ranked_original = toy_service.rerank_candidates(
        candidates,
        regression_model=model,
        feature_columns=feature_columns,
    )
    ranked_loaded = toy_service.rerank_candidates(
        candidates,
        regression_model=loaded,
        feature_columns=feature_columns,
    )

    np.testing.assert_allclose(
        ranked_original["predicted_rating"].to_numpy(dtype=float),
        ranked_loaded["predicted_rating"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        ranked_original["combined_score"].to_numpy(dtype=float),
        ranked_loaded["combined_score"].to_numpy(dtype=float),
    )


def test_load_regression_model_returns_none_when_artifact_missing(tmp_path: Path) -> None:
    assert RetrievalService.load_regression_model(tmp_path / "missing_regressor.joblib") is None


def test_end_to_end_optional_rerank_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRecipeEncoder:
        def __init__(self, config: object) -> None:
            self.config = config

        def load(self) -> None:
            return None

        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            _ = show_progress
            vectors: list[np.ndarray] = []
            for text in texts:
                if "alt" in text.lower():
                    vectors.append(np.array([0.0, 1.0], dtype=float))
                else:
                    vectors.append(np.array([1.0, 0.0], dtype=float))
            return np.vstack(vectors)

    monkeypatch.setattr("recipe_discovery.retrieval.service.RecipeEncoder", FakeRecipeEncoder)

    processed = pd.DataFrame(
        {
            "recipe_id": ["10", "20", "30", "40"],
            "name": ["A", "B", "C", "D"],
            "description": ["d1", "d2", "d3", "d4"],
            "ingredients": ["i1", "i2", "i3", "i4"],
            "steps": ["s1", "s2", "s3", "s4"],
            "minutes": [10.0, 20.0, 35.0, 45.0],
            "n_steps": [1.0, 2.0, 3.0, 4.0],
            "n_ingredients": [3.0, 5.0, 7.0, 4.0],
            "vegetarian": [1, 0, 1, 0],
        }
    )
    processed_path = tmp_path / "processed_subset.csv"
    processed.to_csv(processed_path, index=False)

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.7, 0.3],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    embeddings_path = save_embeddings(embeddings, tmp_path / "recipe_embeddings.npy")
    recipe_ids_path = save_recipe_ids(
        pd.Series(["10", "20", "30", "40"]),
        tmp_path / "recipe_ids.csv",
    )

    service = RetrievalService()
    service.load(
        processed_path=processed_path,
        embeddings_path=embeddings_path,
        recipe_ids_path=recipe_ids_path,
    )

    request = RetrievalRequest(query="quick dinner", top_k=3)
    candidates = service.search(request)
    assert not candidates.empty
    assert "similarity_score" in candidates.columns

    feature_columns = ["minutes", "n_ingredients"]
    model = _train_tiny_regressor(processed, feature_columns)
    model_path = tmp_path / "regressor.joblib"
    model.save(model_path)

    reranked = service.search_with_optional_rerank(
        request,
        regression_model_path=model_path,
        feature_columns=feature_columns,
        similarity_weight=0.7,
        rating_weight=0.3,
    )

    assert not reranked.empty
    assert len(reranked) <= request.top_k
    assert {"similarity_score", "predicted_rating", "combined_score"}.issubset(
        set(reranked.columns)
    )


def test_direct_cosine_and_saved_index_paths_are_consistent(tmp_path: Path) -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.3, 0.1],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=float)

    scores = cosine_similarity(query, embeddings)
    cosine_ranked = np.argsort(-scores)[:3]

    index = build_index(embeddings, n_neighbors=3)
    index_path = save_index(index, tmp_path / "recipe_index.joblib")
    loaded_index = load_index(index_path)
    _, neighbor_idx = loaded_index.kneighbors(query.reshape(1, -1), n_neighbors=3)

    assert neighbor_idx[0].tolist() == cosine_ranked.tolist()


@pytest.fixture
def feedback_service() -> RetrievalService:
    """Create a tiny loaded service for relevance-feedback tests."""

    class DummyEncoder:
        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            _ = show_progress
            return np.array([[1.0, 0.0] for _ in texts], dtype=float)

    service = RetrievalService()
    service.encoder = DummyEncoder()
    service.embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    service.metadata = pd.DataFrame(
        {
            "recipe_id": ["1", "2", "3", "4"],
            "name": ["A", "B", "C", "D"],
            "minutes": [10, 20, 30, 40],
            "n_ingredients": [3, 4, 5, 6],
        }
    )
    service.one_hot_tag_columns = []
    service._attach_foodcom_images = lambda df: df
    return service


def test_encode_text_query_returns_normalized_vector(
    feedback_service: RetrievalService,
) -> None:
    vec = feedback_service.encode_text_query("quick dinner")

    assert vec.shape == (2,)
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_encode_text_query_requires_loaded_encoder() -> None:
    service = RetrievalService()

    with pytest.raises(RuntimeError, match="not loaded"):
        service.encode_text_query("quick dinner")


def test_feedback_excludes_negative_ids(feedback_service: RetrievalService) -> None:
    request = RetrievalRequest(query="quick dinner", top_k=2)
    query_vec = feedback_service.encode_text_query("quick dinner")

    results = feedback_service.search_with_negative_feedback(
        request,
        query_vec=query_vec,
        negative_recipe_ids={"1"},
    )

    assert "1" not in results["recipe_id"].astype(str).tolist()
    assert len(results) <= 2


def test_feedback_rejects_non_text_embedding_space(
    feedback_service: RetrievalService,
) -> None:
    request = RetrievalRequest(query="quick dinner", top_k=2)
    query_vec = feedback_service.encode_text_query("quick dinner")

    with pytest.raises(ValueError, match="Only text feedback"):
        feedback_service.search_with_negative_feedback(
            request,
            query_vec=query_vec,
            negative_recipe_ids={"1"},
            embedding_space="siglip",
        )


def test_feedback_with_unknown_ids_falls_back_to_search(
    feedback_service: RetrievalService,
) -> None:
    request = RetrievalRequest(query="quick dinner", top_k=2)
    query_vec = feedback_service.encode_text_query("quick dinner")

    feedback_results = feedback_service.search_with_negative_feedback(
        request,
        query_vec=query_vec,
        negative_recipe_ids={"999"},
    )
    normal_results = feedback_service.search(request)

    assert feedback_results["recipe_id"].tolist() == normal_results["recipe_id"].tolist()


def test_cosine_similarity_shape() -> None:
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = cosine_similarity(query, matrix)
    assert scores.shape == (2,)


def test_cosine_similarity_raises_on_dimension_mismatch() -> None:
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        cosine_similarity(query, matrix)
