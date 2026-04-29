"""Tests for recipe card parsing helpers."""

from __future__ import annotations

import math

from app.components.recipe_cards import parse_ingredients, parse_reviews, parse_steps


def test_parse_ingredients_supports_python_list_literal() -> None:
    raw = "['flour', 'salt', 'olive oil']"

    assert parse_ingredients(raw) == ["flour", "salt", "olive oil"]


def test_parse_steps_supports_json_list_literal() -> None:
    raw = '["mix dry ingredients", "bake for 20 minutes"]'

    assert parse_steps(raw) == ["mix dry ingredients", "bake for 20 minutes"]


def test_parse_ingredients_supports_pipe_and_comma_fallbacks() -> None:
    assert parse_ingredients("salt|pepper|garlic") == ["salt", "pepper", "garlic"]
    assert parse_ingredients("salt, pepper, garlic") == ["salt", "pepper", "garlic"]


def test_parse_steps_keeps_plain_comma_text_together() -> None:
    raw = "heat oil, add onions, simmer until soft"

    assert parse_steps(raw) == [raw]


def test_parse_helpers_return_empty_for_missing_or_malformed_values() -> None:
    assert parse_ingredients(None) == []
    assert parse_ingredients(float("nan")) == []
    assert parse_steps("[this is not valid list") == []


def test_parse_ingredients_supports_preparsed_iterables() -> None:
    assert parse_ingredients(["milk", "eggs", "butter"]) == ["milk", "eggs", "butter"]
    assert parse_ingredients(("milk", "eggs")) == ["milk", "eggs"]


def test_parse_steps_handles_nan_float() -> None:
    assert parse_steps(math.nan) == []


def test_parse_reviews_supports_serialized_review_list() -> None:
    raw = "['Great weeknight dinner.', 'Would make again.']"

    assert parse_reviews(raw) == ["Great weeknight dinner.", "Would make again."]
