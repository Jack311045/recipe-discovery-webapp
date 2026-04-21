"""Negative-path tests for missing files and bad paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.embeddings.index import load_index
from recipe_discovery.embeddings.store import load_embeddings, load_recipe_ids
from recipe_discovery.retrieval.service import RetrievalService


def test_load_processed_recipes_missing_path_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_processed.csv"
    with pytest.raises(FileNotFoundError, match="Processed CSV not found"):
        load_processed_recipes(missing)



def test_load_embeddings_missing_path_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_embeddings.npy"
    with pytest.raises(FileNotFoundError, match="missing_embeddings"):
        load_embeddings(missing)



def test_load_recipe_ids_missing_path_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_recipe_ids.csv"
    with pytest.raises(FileNotFoundError, match="missing_recipe_ids"):
        load_recipe_ids(missing)



def test_load_index_missing_path_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_index.joblib"
    with pytest.raises(FileNotFoundError, match="Index file not found"):
        load_index(missing)



def test_retrieval_service_load_propagates_missing_processed_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_msg = "Processed CSV not found: data/processed/Processed_data_updated2.csv"

    def _raise_missing(path: Path | None = None) -> None:
        _ = path
        raise FileNotFoundError(missing_msg)

    monkeypatch.setattr("recipe_discovery.retrieval.service.load_processed_recipes", _raise_missing)

    service = RetrievalService()
    with pytest.raises(FileNotFoundError, match="Processed CSV not found"):
        service.load()
