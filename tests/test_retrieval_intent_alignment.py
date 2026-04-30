"""Intent-alignment tests for retrieval quality improvements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recipe_discovery.retrieval.intent import understand_query_intent
from recipe_discovery.retrieval.service import RetrievalRequest, RetrievalService


@pytest.fixture
def intent_alignment_service() -> RetrievalService:
    """Build a deterministic service where dense similarity alone is misleading."""

    class DummyEncoder:
        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            _ = (texts, show_progress)
            return np.tile(np.array([[1.0, 0.0]], dtype=float), (len(texts), 1))

    service = RetrievalService()
    service.encoder = DummyEncoder()
    service.embeddings = np.array(
        [
            [1.0, 0.0],
            [0.97, 0.03],
            [0.95, 0.05],
            [0.93, 0.07],
            [0.9, 0.1],
        ],
        dtype=float,
    )
    service.metadata = pd.DataFrame(
        {
            "recipe_id": ["1", "2", "3", "4", "5"],
            "name": [
                "deep fried burger platter",
                "healthy low fat dinner salad",
                "high protein quick chicken bowl",
                "vegan lunch wrap",
                "creamy comfort pasta",
            ],
            "description": [
                "rich indulgent comfort food with crispy fried sides",
                "light and healthy low-fat dinner option",
                "protein rich quick lunch meal",
                "easy vegan lunch with fresh vegetables",
                "decadent creamy comfort food pasta",
            ],
            "ingredients": [
                "beef|butter|cream|buns",
                "lettuce|grilled chicken|lemon|olive oil",
                "chicken|quinoa|beans|spinach",
                "tortilla|chickpeas|lettuce|tomato",
                "pasta|cream|cheese|butter",
            ],
            "steps": [
                "deep fry and assemble",
                "mix and serve",
                "cook quickly and bowl",
                "wrap and serve",
                "boil and simmer",
            ],
            "minutes": [45, 20, 15, 12, 35],
            "n_ingredients": [10, 9, 11, 8, 12],
            "calories": [920, 320, 430, 390, 710],
            "total fat": [54, 8, 14, 11, 33],
            "sugar": [9, 5, 4, 6, 10],
            "sodium": [1700, 380, 520, 480, 910],
            "protein": [24, 22, 46, 16, 18],
            "saturated fat": [22, 2, 3, 2, 15],
            "carbohydrates": [76, 18, 34, 50, 82],
            "vegan": [0, 0, 0, 1, 0],
            "lunch": [1, 1, 1, 1, 0],
            "main-dish": [1, 1, 1, 1, 1],
            "healthy": [0, 1, 1, 1, 0],
            "low-fat": [0, 1, 0, 1, 0],
            "comfort-food": [1, 0, 0, 0, 1],
            "quick": [0, 1, 1, 1, 0],
            "high-protein": [0, 0, 1, 0, 0],
        }
    )
    service.one_hot_tag_columns = [
        "vegan",
        "lunch",
        "main-dish",
        "healthy",
        "low-fat",
        "comfort-food",
        "quick",
        "high-protein",
    ]
    service._attach_foodcom_images = lambda df: df  # type: ignore[method-assign]
    return service


def test_query_intent_parser_extracts_health_speed_and_tag_signals() -> None:
    intent = understand_query_intent(
        "healthy low fat quick vegan lunch",
        available_tags=["vegan", "lunch", "low-fat", "quick"],
    )

    assert intent.speed_target == "quick"
    assert intent.nutrition_targets.get("total fat") == "low"
    assert "vegan" in intent.preferred_tags
    assert "lunch" in intent.preferred_tags
    assert intent.rewritten_query


def test_contradictory_high_fat_result_gets_penalized(intent_alignment_service: RetrievalService) -> None:
    result = intent_alignment_service.search(
        RetrievalRequest(query="healthy low fat dinner", top_k=5)
    )

    assert result.iloc[0]["recipe_id"] == "2"
    top_contradiction = float(
        result.loc[result["recipe_id"] == "1", "intent_contradiction_penalty"].iloc[0]
    )
    low_fat_contradiction = float(
        result.loc[result["recipe_id"] == "2", "intent_contradiction_penalty"].iloc[0]
    )
    assert top_contradiction > low_fat_contradiction


@pytest.mark.parametrize(
    "query,expected_top_recipe_id",
    [
        ("healthy low fat dinner", "2"),
        ("high protein quick lunch", "3"),
        ("vegan lunch ideas", "4"),
        ("indulgent comfort food", "1"),
    ],
)
def test_intent_alignment_eval_queries(
    intent_alignment_service: RetrievalService,
    query: str,
    expected_top_recipe_id: str,
) -> None:
    result = intent_alignment_service.search(RetrievalRequest(query=query, top_k=3))

    assert not result.empty
    assert result.iloc[0]["recipe_id"] == expected_top_recipe_id
    assert "intent_alignment_score" in result.columns
