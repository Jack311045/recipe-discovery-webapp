"""Tests for lightweight shopping-list helper utilities."""

from __future__ import annotations

from app.components.shopping_list import infer_item_category, merge_ingredients, normalize_ingredient_name


def test_normalize_ingredient_name_cleans_spacing_and_punctuation() -> None:
    assert normalize_ingredient_name("  Olive-oil!  ") == "olive oil"
    assert normalize_ingredient_name("Fresh   Basil") == "fresh basil"


def test_merge_ingredients_deduplicates_batch_and_tracks_source() -> None:
    items: dict[str, dict[str, object]] = {}

    added_count, merged_count = merge_ingredients(
        items,
        ["onion", " onion ", "garlic"],
        source_recipe="Weeknight Pasta",
    )

    assert added_count == 2
    assert merged_count == 0
    assert set(items.keys()) == {"onion", "garlic"}
    assert items["onion"]["source_recipes"] == ["Weeknight Pasta"]


def test_merge_ingredients_handles_simple_plural_merge() -> None:
    items: dict[str, dict[str, object]] = {}

    merge_ingredients(items, ["onion"], source_recipe="Recipe A")
    added_count, merged_count = merge_ingredients(
        items,
        ["onions"],
        source_recipe="Recipe B",
    )

    assert added_count == 0
    assert merged_count == 1
    assert set(items.keys()) == {"onion"}
    assert items["onion"]["source_recipes"] == ["Recipe A", "Recipe B"]


def test_infer_item_category_classifies_known_terms() -> None:
    assert infer_item_category("chicken breast") == "protein"
    assert infer_item_category("olive oil") == "pantry"
    assert infer_item_category("spinach") == "produce"
    assert infer_item_category("mystery ingredient") == "other"
