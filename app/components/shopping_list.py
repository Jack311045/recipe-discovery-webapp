"""Session-state shopping list helpers for a lightweight v1 experience."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

import streamlit as st

SHOPPING_LIST_STATE_KEY = "shopping_list_items"


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
        "pepper",
        "potato",
        "spinach",
        "tomato",
        "zucchini",
    },
    "protein": {
        "bacon",
        "beef",
        "chicken",
        "egg",
        "ham",
        "lamb",
        "pork",
        "sausage",
        "tofu",
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
    "dairy": {
        "butter",
        "cheese",
        "cream",
        "milk",
        "yogurt",
    },
    "pantry": {
        "baking powder",
        "baking soda",
        "black pepper",
        "broth",
        "cinnamon",
        "flour",
        "honey",
        "oil",
        "paprika",
        "pasta",
        "rice",
        "salt",
        "sauce",
        "soy",
        "sugar",
        "vinegar",
    },
}


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


def infer_item_category(normalized_name: str) -> str:
    """Assign a lightweight category for optional grouped display."""
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in normalized_name for keyword in keywords):
            return category
    return "other"


def _clean_source_recipes(raw_sources: object) -> list[str]:
    if not isinstance(raw_sources, list):
        return []

    cleaned: list[str] = []
    for source in raw_sources:
        source_text = _clean_display_name(source)
        if source_text and source_text not in cleaned:
            cleaned.append(source_text)
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
    category = str(raw_item.get("category") or "").strip().lower()
    if category not in {*_CATEGORY_KEYWORDS.keys(), "other"}:
        category = infer_item_category(normalized_name)

    return normalized_name, {
        "normalized_name": normalized_name,
        "display_name": display_name or normalized_name,
        "checked": bool(raw_item.get("checked", False)),
        "category": category,
        "source_recipes": _clean_source_recipes(raw_item.get("source_recipes")),
    }


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
        normalized_name = normalize_ingredient_name(display_name)
        if not normalized_name or normalized_name in seen_in_batch:
            continue
        seen_in_batch.add(normalized_name)

        existing_key = _resolve_existing_key(items, normalized_name)
        entry = items.get(existing_key)

        if entry is None:
            entry = {
                "normalized_name": existing_key,
                "display_name": display_name or existing_key,
                "checked": False,
                "category": infer_item_category(existing_key),
                "source_recipes": [],
            }
            items[existing_key] = entry
            added_count += 1
        else:
            merged_count += 1

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
    values = list(items.values())
    values.sort(
        key=lambda item: (
            bool(item.get("checked")),
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
        category = str(item.get("category") or "other")
        groups[category].append(item)

    ordered_labels = ["produce", "protein", "seafood", "dairy", "pantry", "other"]
    result: dict[str, list[dict[str, object]]] = {}
    for category in ordered_labels:
        if category in groups:
            result[category.title()] = groups[category]
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
