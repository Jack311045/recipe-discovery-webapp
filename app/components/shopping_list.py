"""Session-state shopping list helpers for a lightweight v1 experience."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import streamlit as st

SHOPPING_LIST_STATE_KEY = "shopping_list_items"

_ORDERED_CATEGORIES: list[tuple[str, str]] = [
    ("produce", "Produce"),
    ("meat_poultry", "Meat & Poultry"),
    ("seafood", "Seafood"),
    ("dairy_eggs", "Dairy & Eggs"),
    ("grains_pasta", "Grains & Pasta"),
    ("baking", "Baking"),
    ("spices_seasonings", "Spices & Seasonings"),
    ("condiments_sauces", "Condiments & Sauces"),
    ("pantry_staples", "Pantry Staples"),
    ("other", "Other"),
]

_CATEGORY_LABELS = dict(_ORDERED_CATEGORIES)
_LEGACY_CATEGORY_ALIASES = {
    "dairy": "dairy_eggs",
    "pantry": "pantry_staples",
    "protein": "meat_poultry",
}


_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "produce": {
        "apple",
        "avocado",
        "banana",
        "basil",
        "broccoli",
        "cabbage",
        "carrot",
        "cauliflower",
        "celery",
        "cilantro",
        "cucumber",
        "garlic",
        "ginger",
        "kale",
        "lemon",
        "lettuce",
        "lime",
        "mushroom",
        "onion",
        "orange",
        "parsley",
        "potato",
        "spinach",
        "tomato",
        "zucchini",
    },
    "meat_poultry": {
        "bacon",
        "beef",
        "chicken",
        "ham",
        "lamb",
        "pork",
        "sausage",
        "turkey",
    },
    "seafood": {
        "cod",
        "salmon",
        "shrimp",
        "tilapia",
        "trout",
        "tuna",
    },
    "dairy_eggs": {
        "butter",
        "cheese",
        "cream",
        "egg",
        "milk",
        "yogurt",
    },
    "grains_pasta": {
        "basmati rice",
        "jasmine rice",
        "noodle",
        "oat",
        "pasta",
        "quinoa",
        "rice",
    },
    "baking": {
        "baking powder",
        "baking soda",
        "cocoa",
        "cornstarch",
        "flour",
        "vanilla",
        "yeast",
    },
    "spices_seasonings": {
        "cinnamon",
        "cumin",
        "oregano",
        "paprika",
        "pepper",
        "salt",
        "seasoning",
    },
    "condiments_sauces": {
        "ketchup",
        "mayonnaise",
        "mustard",
        "sauce",
        "soy",
        "soy sauce",
        "sugar",
        "vinegar",
    },
    "pantry_staples": {
        "bean",
        "broth",
        "honey",
        "oil",
        "stock",
        "tofu",
        "water",
    },
}

_VEGETABLE_PEPPER_TERMS = {
    "anaheim pepper",
    "bell pepper",
    "chile pepper",
    "chili pepper",
    "green pepper",
    "habanero",
    "jalapeno",
    "orange pepper",
    "poblano",
    "red pepper",
    "serrano",
    "yellow pepper",
}

_BLACK_PEPPER_TERMS = {
    "black pepper",
    "cracked pepper",
    "fresh ground pepper",
    "ground black pepper",
    "ground pepper",
    "pepper",
    "peppercorn",
    "peppercorns",
    "white pepper",
}

_RICE_CANONICAL_ALIASES = {
    "basmati rice",
    "jasmine rice",
    "long grain rice",
    "rice",
    "white rice",
}

_RICE_KEEP_SEPARATE = {
    "arborio rice",
    "brown rice",
    "rice flour",
    "rice noodle",
    "rice noodles",
    "rice vinegar",
    "sticky rice",
    "sushi rice",
    "wild rice",
}


@dataclass(frozen=True)
class CanonicalIngredient:
    """Normalized ingredient identity used for smart shopping-list merges."""

    canonical_key: str
    display_name: str
    category: str
    merged_variant: str


def ensure_shopping_list_state() -> None:
    """Ensure shopping list state exists in Streamlit session state."""
    if SHOPPING_LIST_STATE_KEY not in st.session_state:
        st.session_state[SHOPPING_LIST_STATE_KEY] = {}


def normalize_ingredient_name(raw_name: object) -> str:
    """Normalize ingredient text for stable deduplication keys."""
    if raw_name is None:
        return ""

    text = str(raw_name).strip().lower()
    if not text:
        return ""

    text = text.replace("&", " and ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_display_name(raw_name: object) -> str:
    if raw_name is None:
        return ""
    text = re.sub(r"\s+", " ", str(raw_name)).strip()
    return text


def _title_from_key(value: str) -> str:
    return value.replace("_", " ").title()


def _normalize_category_key(category: object) -> str:
    key = str(category or "").strip().lower()
    key = _LEGACY_CATEGORY_ALIASES.get(key, key)
    if key in _CATEGORY_LABELS:
        return key
    return "other"


def get_category_label(category_key: str) -> str:
    """Return the display label for a shopping-list category key."""
    return _CATEGORY_LABELS.get(category_key, _title_from_key(category_key))


def _resolve_existing_key(items: dict[str, dict[str, object]], normalized_name: str) -> str:
    if normalized_name in items:
        return normalized_name

    # Conservative plural handling to merge common one-letter plural variants.
    if (
        normalized_name.endswith("s")
        and len(normalized_name) > 3
        and not normalized_name.endswith("ss")
    ):
        singular = normalized_name[:-1]
        if singular in items:
            return singular

    plural = f"{normalized_name}s"
    if plural in items:
        return plural

    return normalized_name


def _contains_any(normalized_name: str, terms: set[str]) -> bool:
    return any(term in normalized_name for term in terms)


def _is_vegetable_pepper(normalized_name: str) -> bool:
    return _contains_any(normalized_name, _VEGETABLE_PEPPER_TERMS)


def _is_black_pepper(normalized_name: str) -> bool:
    return normalized_name in _BLACK_PEPPER_TERMS or (
        "pepper" in normalized_name and not _is_vegetable_pepper(normalized_name)
    )


def _is_salt_and_pepper(normalized_name: str) -> bool:
    normalized = re.sub(r"\b(black|ground|fresh|cracked)\b", "", normalized_name)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in {"salt and pepper", "pepper and salt", "salt pepper"}


def infer_item_category(normalized_name: str) -> str:
    """Assign a grocery-aisle category key for grouped display."""
    normalized_name = normalize_ingredient_name(normalized_name)
    if not normalized_name:
        return "other"

    if _is_vegetable_pepper(normalized_name):
        return "produce"
    if _is_salt_and_pepper(normalized_name) or _is_black_pepper(normalized_name):
        return "spices_seasonings"
    if normalized_name in _RICE_KEEP_SEPARATE:
        if normalized_name == "rice vinegar":
            return "condiments_sauces"
        if normalized_name == "rice flour":
            return "baking"
        return "grains_pasta"

    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in normalized_name for keyword in keywords):
            return category
    return "other"


def canonicalize_ingredient(
    raw_name: object,
    existing_items: Mapping[str, Mapping[str, object]] | None = None,
) -> CanonicalIngredient | None:
    """Return conservative merge metadata for one ingredient line."""
    display_name = _clean_display_name(raw_name)
    normalized_name = normalize_ingredient_name(display_name)
    if not normalized_name:
        return None

    canonical_key = normalized_name
    canonical_display = display_name or normalized_name

    if _is_salt_and_pepper(normalized_name):
        canonical_key = "salt and pepper"
        canonical_display = "Salt and pepper"
    elif (
        existing_items is not None
        and "salt and pepper" in existing_items
        and (normalized_name == "salt" or _is_black_pepper(normalized_name))
    ):
        canonical_key = "salt and pepper"
        canonical_display = "Salt and pepper"
    elif normalized_name in _RICE_CANONICAL_ALIASES:
        canonical_key = "rice"
        canonical_display = "Rice"

    return CanonicalIngredient(
        canonical_key=canonical_key,
        display_name=canonical_display,
        category=infer_item_category(canonical_key),
        merged_variant=display_name or normalized_name,
    )


def _clean_source_recipes(raw_sources: object) -> list[str]:
    if not isinstance(raw_sources, list):
        return []

    cleaned: list[str] = []
    for source in raw_sources:
        source_text = _clean_display_name(source)
        if source_text and source_text not in cleaned:
            cleaned.append(source_text)
    return cleaned


def _clean_string_list(raw_values: object) -> list[str]:
    if not isinstance(raw_values, list):
        return []

    cleaned: list[str] = []
    for value in raw_values:
        text = _clean_display_name(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _clean_source_ingredients(raw_sources: object) -> dict[str, list[str]]:
    if not isinstance(raw_sources, dict):
        return {}

    cleaned: dict[str, list[str]] = {}
    for source, values in raw_sources.items():
        source_name = _clean_display_name(source)
        if not source_name:
            continue
        lines = _clean_string_list(values)
        if lines:
            cleaned[source_name] = lines
    return cleaned


def _sanitize_shopping_item(
    raw_key: object,
    raw_item: Mapping[str, object],
) -> tuple[str, dict[str, object]] | None:
    normalized_name = normalize_ingredient_name(
        raw_item.get("normalized_name") or raw_key
    )
    if not normalized_name:
        return None

    display_name = _clean_display_name(raw_item.get("display_name") or normalized_name)
    raw_category = str(raw_item.get("category") or "").strip().lower()
    category = _normalize_category_key(raw_category)
    if raw_category in _LEGACY_CATEGORY_ALIASES or category == "other":
        category = infer_item_category(normalized_name)

    return normalized_name, {
        "normalized_name": normalized_name,
        "display_name": display_name or normalized_name,
        "checked": bool(raw_item.get("checked", False)),
        "category": category,
        "source_recipes": _clean_source_recipes(raw_item.get("source_recipes")),
        "source_ingredients": _clean_source_ingredients(
            raw_item.get("source_ingredients")
        ),
        "merged_variants": _clean_string_list(raw_item.get("merged_variants")),
    }


def _append_unique(values: list[object], value: object) -> None:
    text = _clean_display_name(value)
    if text and text not in values:
        values.append(text)


def _record_source_ingredient(
    entry: dict[str, object],
    *,
    source_recipe: str | None,
    ingredient_line: str,
) -> None:
    source_name = _clean_display_name(source_recipe)
    if not source_name:
        return

    raw_mapping = entry.setdefault("source_ingredients", {})
    if not isinstance(raw_mapping, dict):
        raw_mapping = {}
        entry["source_ingredients"] = raw_mapping

    lines = raw_mapping.setdefault(source_name, [])
    if isinstance(lines, list):
        _append_unique(lines, ingredient_line)


def merge_ingredients(
    items: dict[str, dict[str, object]],
    ingredients: Sequence[object],
    *,
    source_recipe: str | None = None,
) -> tuple[int, int]:
    """Merge a batch of ingredient lines into an existing shopping list map."""
    added_count = 0
    merged_count = 0
    seen_in_batch: set[str] = set()

    for ingredient in ingredients:
        display_name = _clean_display_name(ingredient)
        canonical = canonicalize_ingredient(display_name, existing_items=items)
        if canonical is None or canonical.canonical_key in seen_in_batch:
            continue
        seen_in_batch.add(canonical.canonical_key)

        existing_key = _resolve_existing_key(items, canonical.canonical_key)
        entry = items.get(existing_key)

        if entry is None:
            entry = {
                "normalized_name": existing_key,
                "display_name": canonical.display_name,
                "checked": False,
                "category": canonical.category,
                "source_recipes": [],
                "source_ingredients": {},
                "merged_variants": [],
            }
            items[existing_key] = entry
            added_count += 1
        else:
            current_category = _normalize_category_key(entry.get("category"))
            entry["category"] = (
                canonical.category
                if canonical.category != "other"
                else current_category
            )
            if "source_ingredients" not in entry or not isinstance(
                entry.get("source_ingredients"), dict
            ):
                entry["source_ingredients"] = {}
            if "merged_variants" not in entry or not isinstance(
                entry.get("merged_variants"), list
            ):
                entry["merged_variants"] = []
            merged_count += 1

        variants = entry.setdefault("merged_variants", [])
        if isinstance(variants, list):
            _append_unique(variants, canonical.merged_variant)

        _record_source_ingredient(
            entry,
            source_recipe=source_recipe,
            ingredient_line=display_name,
        )

        if source_recipe:
            source_recipes = entry.setdefault("source_recipes", [])
            if isinstance(source_recipes, list) and source_recipe not in source_recipes:
                source_recipes.append(source_recipe)

    return added_count, merged_count


def add_ingredients_to_shopping_list(
    ingredients: Sequence[object],
    *,
    source_recipe: str | None = None,
) -> tuple[int, int]:
    """Merge ingredient lines into shopping list state and return add/merge counts."""
    ensure_shopping_list_state()
    items = st.session_state[SHOPPING_LIST_STATE_KEY]
    return merge_ingredients(items, ingredients, source_recipe=source_recipe)


def add_manual_item_to_shopping_list(item_text: str) -> bool:
    """Add one manually-entered item to the shopping list."""
    added_count, _ = add_ingredients_to_shopping_list([item_text])
    return added_count > 0


def get_shopping_items() -> list[dict[str, object]]:
    """Return sorted shopping list items for rendering."""
    ensure_shopping_list_state()
    items: dict[str, dict[str, object]] = st.session_state[SHOPPING_LIST_STATE_KEY]
    sanitized: dict[str, dict[str, object]] = {}
    for key, item in items.items():
        if isinstance(item, Mapping):
            sanitized_item = _sanitize_shopping_item(key, item)
            if sanitized_item is not None:
                sanitized[sanitized_item[0]] = sanitized_item[1]
    if len(sanitized) != len(items) or sanitized != items:
        st.session_state[SHOPPING_LIST_STATE_KEY] = sanitized
        items = sanitized

    values = list(items.values())
    values.sort(
        key=lambda item: (
            bool(item.get("checked")),
            _normalize_category_key(item.get("category")),
            str(item.get("display_name", "")).lower(),
        )
    )
    return values


def get_grouped_shopping_items(
    group_by_category: bool = True,
) -> dict[str, list[dict[str, object]]]:
    """Return grouped shopping items with predictable category ordering."""
    items = get_shopping_items()
    if not group_by_category:
        return {"All items": items}

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        category = _normalize_category_key(item.get("category"))
        groups[category].append(item)

    result: dict[str, list[dict[str, object]]] = {}
    for category, label in _ORDERED_CATEGORIES:
        if category in groups:
            category_items = groups[category]
            result[f"{label} ({len(category_items)})"] = category_items
    return result


def set_item_checked(normalized_name: str, checked: bool) -> None:
    """Update checked status for one shopping list item."""
    ensure_shopping_list_state()
    items: dict[str, dict[str, object]] = st.session_state[SHOPPING_LIST_STATE_KEY]
    if normalized_name in items:
        items[normalized_name]["checked"] = bool(checked)


def remove_shopping_item(normalized_name: str) -> None:
    """Remove one item from shopping list state."""
    ensure_shopping_list_state()
    items: dict[str, dict[str, object]] = st.session_state[SHOPPING_LIST_STATE_KEY]
    items.pop(normalized_name, None)


def clear_shopping_list() -> None:
    """Remove all items from shopping list state."""
    st.session_state[SHOPPING_LIST_STATE_KEY] = {}


def remove_checked_items() -> int:
    """Remove all checked items and return removed count."""
    ensure_shopping_list_state()
    items: dict[str, dict[str, object]] = st.session_state[SHOPPING_LIST_STATE_KEY]
    to_remove = [key for key, value in items.items() if bool(value.get("checked"))]
    for key in to_remove:
        items.pop(key, None)
    return len(to_remove)


def get_shopping_list_count() -> int:
    """Return current shopping list item count."""
    ensure_shopping_list_state()
    items: dict[str, dict[str, object]] = st.session_state[SHOPPING_LIST_STATE_KEY]
    return len(items)


def get_checked_item_count() -> int:
    """Return number of checked/completed shopping list items."""
    ensure_shopping_list_state()
    items: dict[str, dict[str, object]] = st.session_state[SHOPPING_LIST_STATE_KEY]
    return sum(1 for value in items.values() if bool(value.get("checked")))
