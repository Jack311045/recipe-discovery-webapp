"""Tests for lightweight shopping-list helper utilities."""

from __future__ import annotations

from app.components.shopping_list import (
    canonicalize_ingredient,
    infer_item_category,
    merge_ingredients,
    normalize_ingredient_name,
)


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


def test_salt_and_pepper_absorbs_standalone_salt_with_sources() -> None:
    items: dict[str, dict[str, object]] = {}

    merge_ingredients(items, ["salt and pepper"], source_recipe="Recipe A")
    added_count, merged_count = merge_ingredients(
        items,
        ["salt"],
        source_recipe="Recipe B",
    )

    assert added_count == 0
    assert merged_count == 1
    assert set(items.keys()) == {"salt and pepper"}
    assert items["salt and pepper"]["source_recipes"] == ["Recipe A", "Recipe B"]
    assert items["salt and pepper"]["source_ingredients"] == {
        "Recipe A": ["salt and pepper"],
        "Recipe B": ["salt"],
    }
    assert items["salt and pepper"]["merged_variants"] == ["salt and pepper", "salt"]


def test_salt_and_black_pepper_canonicalizes_to_salt_and_pepper() -> None:
    canonical = canonicalize_ingredient("salt and black pepper")

    assert canonical is not None
    assert canonical.canonical_key == "salt and pepper"
    assert canonical.display_name == "Salt and pepper"
    assert canonical.category == "spices_seasonings"


def test_jasmine_and_basmati_rice_merge_to_generic_rice() -> None:
    items: dict[str, dict[str, object]] = {}

    merge_ingredients(items, ["jasmine rice"], source_recipe="Recipe A")
    added_count, merged_count = merge_ingredients(
        items,
        ["basmati rice"],
        source_recipe="Recipe B",
    )

    assert added_count == 0
    assert merged_count == 1
    assert set(items.keys()) == {"rice"}
    assert items["rice"]["display_name"] == "Rice"
    assert items["rice"]["category"] == "grains_pasta"
    assert items["rice"]["source_recipes"] == ["Recipe A", "Recipe B"]
    assert items["rice"]["merged_variants"] == ["jasmine rice", "basmati rice"]


def test_merge_refreshes_legacy_category_for_existing_item() -> None:
    items: dict[str, dict[str, object]] = {
        "rice": {
            "normalized_name": "rice",
            "display_name": "rice",
            "checked": False,
            "category": "pantry",
            "source_recipes": [],
        }
    }

    added_count, merged_count = merge_ingredients(
        items,
        ["basmati rice"],
        source_recipe="Recipe B",
    )

    assert added_count == 0
    assert merged_count == 1
    assert items["rice"]["category"] == "grains_pasta"


def test_specific_rice_products_do_not_merge_to_generic_rice() -> None:
    items: dict[str, dict[str, object]] = {}

    added_count, merged_count = merge_ingredients(
        items,
        ["brown rice", "rice vinegar", "rice noodles"],
        source_recipe="Recipe A",
    )

    assert added_count == 3
    assert merged_count == 0
    assert set(items.keys()) == {"brown rice", "rice vinegar", "rice noodles"}


def test_black_pepper_does_not_merge_with_bell_pepper() -> None:
    items: dict[str, dict[str, object]] = {}

    merge_ingredients(items, ["black pepper"], source_recipe="Recipe A")
    added_count, merged_count = merge_ingredients(
        items,
        ["bell pepper"],
        source_recipe="Recipe B",
    )

    assert added_count == 1
    assert merged_count == 0
    assert set(items.keys()) == {"black pepper", "bell pepper"}
    assert items["black pepper"]["category"] == "spices_seasonings"
    assert items["bell pepper"]["category"] == "produce"


def test_source_ingredients_records_original_lines_per_recipe() -> None:
    items: dict[str, dict[str, object]] = {}

    merge_ingredients(items, ["jasmine rice"], source_recipe="Recipe A")
    merge_ingredients(items, ["basmati rice"], source_recipe="Recipe B")

    assert items["rice"]["source_ingredients"] == {
        "Recipe A": ["jasmine rice"],
        "Recipe B": ["basmati rice"],
    }


def test_infer_item_category_classifies_known_terms() -> None:
    assert infer_item_category("chicken breast") == "meat_poultry"
    assert infer_item_category("olive oil") == "pantry_staples"
    assert infer_item_category("spinach") == "produce"
    assert infer_item_category("rice") == "grains_pasta"
    assert infer_item_category("salt and pepper") == "spices_seasonings"
    assert infer_item_category("mystery ingredient") == "other"
